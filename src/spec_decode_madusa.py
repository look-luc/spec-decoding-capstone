"""
Speculative decoding implementation.

Contains:
- speculative_decode_greedy: Custom greedy speculative decoding with KV caching
- get_stop_token_ids: Stop token detection for various chat models
- crop_kv_cache: KV cache management utility
"""

import time
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from src.models import madusa


def sample(logprobs: torch.Tensor, mode: Literal["greedy", "sample"]):
    """Sample a token index from (already filtered) log-probs."""
    if mode == "greedy":
        return logprobs.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logprobs).sample()

def filter_logprobs(
    logprobs: torch.Tensor, top_k: int = 0, top_p: float = 0.0
) -> torch.Tensor:
    """Apply top-k and/or top-p filtering, then renormalize to valid log-probs."""
    filtered = logprobs
    if top_k > 0:
        filtered = apply_top_k(filtered, k=top_k)
    if 0.0 < top_p < 1.0:
        filtered = apply_top_p(filtered, p=top_p)
    if top_k > 0 or 0.0 < top_p < 1.0:
        filtered = torch.log_softmax(filtered, dim=-1)
    return filtered

def get_stop_token_ids(tokenizer, eos_token_id=None):
    """
    Get all stop token IDs for chat models.
    Supports: Qwen, Llama, Mistral, Gemma, and others.
    """
    stop_ids = set()

    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    stop_tokens = [
        "<|im_end|>",  # Qwen
        "<|endoftext|>",  # Qwen, GPT
        "<|eot_id|>",  # Llama 3
        "<|end_of_text|>",  # Llama 3
        "</s>",  # Mistral, Llama 2
        "<end_of_turn>",  # Gemma
        "<eos>",  # Gemma
        "[/INST]",  # Mistral
    ]

    for token in stop_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids and len(ids) == 1:
                stop_ids.add(ids[0])
        except Exception:
            pass

    return stop_ids


def crop_kv_cache(
    past_key_values,
    prefix_len: int,
    best_path: list | None = None,
    max_accept_len: int = 0,
    tree_node_dict: dict | None = None,
):
    """
    Crop and gather KV cache entries for Medusa tree speculative decoding.

    Avoids direct iteration over past_key_values to prevent Cache.__iter__
    failures on LinearAttentionLayer objects.
    """
    if past_key_values is None:
        return None

    # Construct sequence indices to keep
    keep_indices = list(range(prefix_len))
    if max_accept_len > 0 and best_path is not None and tree_node_dict is not None:
        for depth in range(max_accept_len):
            path_prefix = tuple(best_path[: depth + 1])
            node_idx = tree_node_dict[path_prefix]
            keep_indices.append(prefix_len + node_idx)

    # Standard Hugging Face DynamicCache (key_cache and value_cache attributes)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        if len(past_key_values.key_cache) > 0:
            device = past_key_values.key_cache[0].device
            idx = torch.tensor(keep_indices, dtype=torch.long, device=device)
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = torch.index_select(
                    past_key_values.key_cache[i], dim=2, index=idx
                )
                past_key_values.value_cache[i] = torch.index_select(
                    past_key_values.value_cache[i], dim=2, index=idx
                )
        return past_key_values

    # Newer Hugging Face Cache implementations using .layers
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and getattr(layer, "keys", None) is not None:
                idx = torch.tensor(keep_indices, dtype=torch.long, device=layer.keys.device)
                layer.keys = torch.index_select(layer.keys, dim=2, index=idx)
                layer.values = torch.index_select(layer.values, dim=2, index=idx)
            elif hasattr(layer, "key_states") and getattr(layer, "key_states", None) is not None:
                idx = torch.tensor(keep_indices, dtype=torch.long, device=layer.key_states.device)
                layer.key_states = torch.index_select(layer.key_states, dim=2, index=idx)
                layer.value_states = torch.index_select(layer.value_states, dim=2, index=idx)
        return past_key_values

    # Legacy tuple/list cache format ((key_state, value_state), ...)
    if isinstance(past_key_values, (tuple, list)):
        new_past = []
        for layer_past in past_key_values:
            if isinstance(layer_past, torch.Tensor):
                idx = torch.tensor(keep_indices, dtype=torch.long, device=layer_past.device)
                new_past.append(torch.index_select(layer_past, dim=-1, index=idx))
            elif isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                k_state, v_state = layer_past
                idx = torch.tensor(keep_indices, dtype=torch.long, device=k_state.device)
                k_cropped = torch.index_select(k_state, dim=2, index=idx)
                v_cropped = torch.index_select(v_state, dim=2, index=idx)
                new_past.append((k_cropped, v_cropped))
        return tuple(new_past)

    return past_key_values

def get_kv_cache_length(past_key_values) -> int:
    """Helper to get the current sequence length of a KV cache."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        if len(past_key_values[0][0].shape) == 4:
            # HF cache
            return past_key_values[0][0].size(2)
        else:
            return past_key_values[0][0].size(-1)
    return 0

def default_tree(preset_type: str, num_heads: int = 4):
    preset_type = preset_type.lower()
    if preset_type not in ["lightweight", "greedy_linear", "standard", "deep_linear"]:
        raise ValueError(f"Unknown preset: {preset_type}")

    if preset_type == "greedy_linear" or preset_type == "deep_linear":
        return [[0] * (depth + 1) for depth in range(num_heads)]

    if preset_type == "lightweight":
        return [
            [0], [0, 0], [0, 0, 0], [0, 0, 0, 0],
            [1], [0, 1], [1, 0], [0, 0, 1],
            [2], [0, 2], [2, 0], [0, 1, 0],
            [3], [0, 0, 2], [1, 1], [0, 0, 0, 1]
        ]
    elif preset_type == "standard":
        return [
            [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3],
            [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0],
            [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0],
            [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1],
            [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0],
            [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0],
            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6],
            [0, 7, 0]
        ]

def build_tree(
    logits,
    tree_choice,
    cur_gen_idx,
    past_kv_len,
    top_k,
    top_p,
    mode
):
    if logits.ndim == 3:
        logits = logits.squeeze(1)

    num_heads = logits.size(0)

    valid_tree_choice = []
    for path in tree_choice:
        if len(path) <= num_heads:
            valid_tree_choice.append(path)
        else:
            valid_tree_choice.append(path[:num_heads])

    max_rank_per_depth = [0] * num_heads
    for path in valid_tree_choice:
        for depth, rank in enumerate(path):
            if rank > max_rank_per_depth[depth]:
                max_rank_per_depth[depth] = rank

    top_token_per_head = []
    for depth in range(num_heads):
        max_rank = max_rank_per_depth[depth]
        head_logits = filter_logprobs(
            F.log_softmax(logits[depth], dim=-1),
            top_k=top_k,
            top_p=top_p
        )
        top_ids = torch.topk(head_logits, k=max_rank + 1).indices
        top_token_per_head.append(top_ids)

    nodes = []
    node_dict = {}

    for path in valid_tree_choice:
        for depth in range(len(path)):
            rank = path[depth]
            token_id = top_token_per_head[depth][rank]

            path_prefix = tuple(path[: depth + 1])
            parent_prefix = tuple(path[: depth])

            if path_prefix not in node_dict:
                new_node_idx = len(nodes)
                node_dict[path_prefix] = new_node_idx

                parent_idx = node_dict[parent_prefix] if len(parent_prefix) > 0 else None

                nodes.append(
                    {
                        "node_idx": new_node_idx,
                        "token_id": token_id,
                        "depth": depth,
                        "parent_idx": parent_idx,
                    }
                )

    size = len(nodes)
    draft_tree_tokens = torch.zeros((1, size), dtype=torch.long, device=logits.device)
    pos_idx = torch.zeros((1, size), dtype=torch.long, device=logits.device)

    for idx in range(size):
        draft_tree_tokens[0, idx] = nodes[idx]["token_id"]
        pos_idx[0, idx] = cur_gen_idx + nodes[idx]["depth"]

    attn_mask = torch.full(
        size=(1, 1, size, past_kv_len + size),
        fill_value=-float('inf'),
        device=logits.device,
        dtype=logits.dtype
    )
    attn_mask[:, :, :, :past_kv_len] = 0.0

    for i in range(size):
        cur_node = nodes[i]
        while cur_node is not None:
            attn_mask[0, 0, i, past_kv_len + cur_node["node_idx"]] = 0.0
            cur_node = nodes[cur_node["parent_idx"]] if cur_node["parent_idx"] is not None else None

    return {
        "tokens": draft_tree_tokens,
        "attention": attn_mask,
        "pos_idx": pos_idx,
        "paths": valid_tree_choice,
        "nodes": nodes,
        "node_dict": node_dict,
    }

def speculative_decode(
    target_model,
    tokenizer,
    input_ids,
    mode: Literal["greedy", "sample"],
    medusa: nn.Module,
    num_heads:int,
    max_new_tokens=128,
    tree_choices: str | list[list] = "Standard",
    top_k=0,
    top_p=0.0,
    repetition_penalty=1.1,
    repetition_penalty_window=16,
    eos_token_id=None,
    device=None,
    track_iterations: bool=False,
):
    """
    Medusa Speculative Decoding with KV Caching.
    """
    bs = input_ids.size(0)
    assert bs == 1, "Speculative decoding only supports batch_size=1"

    if device is None:
        device = next(target_model.parameters()).device

    if medusa is None:
        medusa = madusa.madusa(base_model=target_model, num_heads=num_heads)
    medusa = medusa.to(device=device, dtype=target_model.dtype)

    if isinstance(tree_choices, str):
        tree_choices = default_tree(tree_choices, num_heads=num_heads)

    def apply_filters(logprobs: torch.Tensor) -> torch.Tensor:
        return filter_logprobs(logprobs, top_k=top_k, top_p=top_p)

    def select_index(logprobs: torch.Tensor):
        return sample(logprobs, mode)

    def penalize_logits(
        logits: torch.Tensor,
        confirmed_len: int,
        draft_so_far: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if repetition_penalty == 1.0:
            return logits
        ctx = generated_tokens[:, :confirmed_len]
        if draft_so_far is not None and draft_so_far.size(-1) > 0:
            ctx = torch.cat([ctx, draft_so_far], dim=-1)
        ctx = ctx[:, -repetition_penalty_window:]
        return apply_repetition_penalty(logits.clone(), ctx, repetition_penalty)

    stop_token_ids = torch.tensor(
        list(get_stop_token_ids(tokenizer, eos_token_id)), device=device
    )
    input_ids = input_ids.to(device)

    d_vocab = max(medusa.vocab_size, target_model.config.vocab_size)

    generated_tokens = torch.concat(
        [
            input_ids,
            torch.zeros(
                bs, max_new_tokens, device=device, dtype=torch.int64
            ),
        ],
        dim=-1,
    )
    prompt_len = input_ids.size(-1)
    cur_gen_idx = input_ids.size(-1)

    draft_start, draft_end, verifier_start, verifier_end = None, None, None, None
    draft_times_acc = (0.0, 0.0, 0)
    verifier_times_acc = (0.0, 0.0, 0)
    if device.type == "cuda":
        draft_start = torch.cuda.Event(enable_timing=True)
        draft_end = torch.cuda.Event(enable_timing=True)
        verifier_start = torch.cuda.Event(enable_timing=True)
        verifier_end = torch.cuda.Event(enable_timing=True)

    def get_time():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()

    with torch.no_grad():
        # Preload kv cache for prompts
        target_out = target_model(input_ids, use_cache=True, output_hidden_states=True)
        target_kv_cache = target_out.past_key_values
        last_hidden = target_out.hidden_states[-1][:, -1:, :]  # shape [bs, 1, hidden_dim]

        # Add the first new token.
        first_logits = penalize_logits(target_out.logits[:, -1, :], confirmed_len=cur_gen_idx)
        first_target_token = select_index(
            apply_filters(torch.log_softmax(first_logits, dim=-1))
        )
        generated_tokens[:, cur_gen_idx] = first_target_token
        cur_gen_idx += 1

        prev_target_logits = first_logits

        # Metrics
        total_draft_tokens = 0
        total_matched_tokens = 0
        octile_offsets = [i * (max_new_tokens // 8) + 1 for i in range(8)]
        per_position_accept_count = [0] * 8
        per_position_draft_count = [0] * 8
        num_iterations = 0
        iteration_history = []
        start_time = get_time()

        while cur_gen_idx < generated_tokens.size(-1):
            num_iterations += 1
            past_kv_len = get_kv_cache_length(target_kv_cache)
            _ = draft_start and draft_start.record()
            medusa_logits = medusa(hidden_states=last_hidden)

            tree_data = build_tree(
                logits=medusa_logits,
                cur_gen_idx=cur_gen_idx,
                tree_choice=tree_choices,
                past_kv_len=past_kv_len,
                top_k=top_k,
                top_p=top_p,
                mode=mode,
            )
            draft_tree_tokens = tree_data["tokens"]
            tree_atten_mask = tree_data["attention"]
            tree_pos_id = tree_data["pos_idx"]
            tree_paths = tree_data["paths"]
            tree_nodes = tree_data["nodes"]
            tree_node_dict = tree_data["node_dict"]

            _ = draft_end and draft_end.record()

            total_draft_tokens += draft_tree_tokens.size(-1)

            _ = verifier_start and verifier_start.record()
            target_out = target_model(
                input_ids=draft_tree_tokens,
                past_key_values=target_kv_cache,
                attention_mask=tree_atten_mask,
                position_ids=tree_pos_id,
                use_cache=True,
                output_hidden_states=True,
            )
            _ = verifier_end and verifier_end.record()

            if draft_start and draft_end and verifier_start and verifier_end:
                torch.cuda.synchronize()

                draft_elapsed = draft_start.elapsed_time(draft_end)
                n_drafted = draft_tree_tokens.size(-1)
                draft_times_acc = (
                    draft_times_acc[0] + draft_elapsed,
                    draft_times_acc[1] + (draft_elapsed**2 / n_drafted if n_drafted > 0 else 0.0),
                    draft_times_acc[2] + n_drafted,
                )

                verifier_elapsed = verifier_start.elapsed_time(verifier_end)
                verifier_times_acc = (
                    verifier_times_acc[0] + verifier_elapsed,
                    verifier_times_acc[1] + verifier_elapsed**2,
                    verifier_times_acc[2] + 1,
                )

            verify_logits = target_out.logits

            best_path = None
            best_accepted_token = []
            best_bonus_token = None
            best_bonus_logits = None
            max_accept_len = -1

            for path in tree_paths:
                accepted_in_path = []
                bonus_token = None
                bonus_logits = None
                path_matched = True

                for depth in range(len(path)):
                    path_prefix = tuple(path[: depth + 1])
                    node_i = tree_node_dict[path_prefix]
                    draft_token = draft_tree_tokens[:, node_i]

                    if depth == 0:
                        raw_pred_logits = prev_target_logits
                    else:
                        parent_node_idx = tree_nodes[node_i]["parent_idx"]
                        raw_pred_logits = verify_logits[:, parent_node_idx, :]

                    node_raw_logits = penalize_logits(
                        raw_pred_logits,
                        confirmed_len=cur_gen_idx + depth,
                    )
                    target_dist = apply_filters(F.log_softmax(node_raw_logits, dim=-1))
                    verified_token = select_index(target_dist)

                    if draft_token == verified_token:
                        accepted_in_path.append(draft_token)
                    else:
                        bonus_token = verified_token
                        bonus_logits = node_raw_logits
                        path_matched = False
                        break

                if path_matched and bonus_token is None:
                    last_node_idx = tree_node_dict[tuple(path)]
                    bonus_raw_logits = penalize_logits(
                        verify_logits[:, last_node_idx, :],
                        confirmed_len=cur_gen_idx + len(path),
                        draft_so_far=generated_tokens,
                    )
                    bonus_token = select_index(
                        apply_filters(
                            F.log_softmax(bonus_raw_logits, dim=-1)
                        )
                    )
                    bonus_logits = bonus_raw_logits

                if len(accepted_in_path) > max_accept_len:
                    max_accept_len = len(accepted_in_path)
                    best_accepted_token = accepted_in_path
                    best_bonus_token = bonus_token
                    best_bonus_logits = bonus_logits
                    best_path = path

            gen_offset = cur_gen_idx - prompt_len
            draft_depth = len(best_path) if best_path is not None else 0

            for i in range(len(octile_offsets)):
                checkpoint = octile_offsets[i]
                if gen_offset <= checkpoint and checkpoint < (gen_offset + draft_depth):
                    per_position_draft_count[i] = per_position_draft_count[i] + 1
                    rel_depth = checkpoint - gen_offset
                    if rel_depth == checkpoint - gen_offset:
                        per_position_accept_count[i] = per_position_accept_count[i] + 1

            bonus_tok = best_bonus_token.view(-1) if best_bonus_token.ndim > 0 else best_bonus_token.unsqueeze(0)
            accepted_toks = [t.view(-1) for t in best_accepted_token]
            tokens_to_add = torch.cat([*accepted_toks, bonus_tok], dim=-1).unsqueeze(0)
            new_gen_idx = cur_gen_idx + tokens_to_add.size(dim=-1)
            generated_tokens[:, cur_gen_idx:new_gen_idx] = tokens_to_add

            total_matched_tokens += len(best_accepted_token)

            target_kv_cache = crop_kv_cache(
                past_key_values=target_out.past_key_values,
                prefix_len=past_kv_len,
                best_path=best_path,
                max_accept_len=max_accept_len,
                tree_node_dict=tree_node_dict,
            )

            if max_accept_len > 0 and best_path is not None:
                last_accepted_prefix = tuple(best_path[:max_accept_len])
                last_node_idx = tree_node_dict[last_accepted_prefix]
                last_hidden = target_out.hidden_states[-1][:, last_node_idx:last_node_idx + 1, :].detach().clone()
            else:
                last_hidden = last_hidden.detach().clone()

            prev_target_logits = best_bonus_logits
            cur_gen_idx = new_gen_idx

            del target_out

            if generated_tokens[:, cur_gen_idx - 1] in stop_token_ids:
                generated_tokens = generated_tokens[:, :cur_gen_idx]
                break

    total_time = get_time() - start_time
    acceptance_rate = total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0

    octile_position_acceptance = [
        acc / draf if draf > 0 else None for acc, draf in zip(per_position_accept_count, per_position_draft_count)
    ]

    metrics = {
        "time": total_time,
        "generated_tokens": cur_gen_idx - prompt_len,
        "draft_tokens": total_draft_tokens,
        "matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
        "octile_position_acceptance": octile_position_acceptance,
        "octile_positions": octile_offsets,
        "num_iterations": num_iterations,
        "toks_per_sec": (cur_gen_idx - prompt_len) / total_time if total_time > 0 else 0,
    }

    if draft_times_acc[2] > 0 and verifier_times_acc[2] > 0:
        d_sum, d_sum_sq, d_n = draft_times_acc
        v_sum, v_sum_sq, v_n = verifier_times_acc

        average_draft_time = d_sum / d_n
        average_verifier_time = v_sum / v_n

        metrics["average_draft_time"] = average_draft_time / 1000.0
        metrics["average_verifier_time"] = average_verifier_time / 1000.0

        raw_draft_variance = d_sum_sq / d_n - average_draft_time**2
        raw_verifier_variance = v_sum_sq / v_n - average_verifier_time**2

        metrics["draft_time_variance"] = max(raw_draft_variance, 0.0) / 1e6
        metrics["verifier_time_variance"] = max(raw_verifier_variance, 0.0) / 1e6
        metrics["draft_time_count"] = d_n
        metrics["verifier_time_count"] = v_n

    if track_iterations:
        metrics["iteration_history"] = iteration_history

    return generated_tokens, metrics

def apply_repetition_penalty(
    logits: torch.Tensor,
    context_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply multiplicative repetition penalty to raw logits (single position)."""
    if penalty == 1.0 or context_ids.size(-1) == 0:
        return logits

    for b in range(logits.size(0)):
        max_vocab_index = torch.max(context_ids[b]).item() + 1
        max_vocab_index = cast(int, max_vocab_index)
        window_counts = torch.zeros(max_vocab_index, dtype=torch.long, device=context_ids[b].device)
        window_counts.scatter_add_(dim=0, index=context_ids[b], src=torch.ones_like(context_ids[b]))

        per_token_penalty = penalty ** window_counts
        logits[b,:max_vocab_index] = torch.where(
            logits[b,:max_vocab_index] > 0,
            logits[b,:max_vocab_index] / per_token_penalty,
            logits[b,:max_vocab_index] * per_token_penalty,
        )
    return logits


def apply_repetition_penalty_batched(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    confirmed_len: int,
    penalty: float,
    window: int,
) -> torch.Tensor:
    """Vectorized repetition penalty for all verification positions at once.

    Position j's context = generated_tokens[j-window:j]
    """
    if penalty == 1.0:
        return logits

    bs, seq_len = generated_tokens.shape
    device = generated_tokens.device

    for b in range(bs):
        # Only positions before the current pos
        mask = ~torch.triu(torch.ones(seq_len + 1, seq_len, dtype=torch.bool, device=device))

        # Only positions after the start of the window
        window_start = torch.clamp(torch.arange(seq_len + 1, device=device) - window, 0).unsqueeze(-1)
        start_mask = torch.arange(seq_len, device=device) >= window_start
        mask *= start_mask

        # Replace masked positions with an unused index (hack to avoid using 0)
        unused_idx = torch.max(generated_tokens).item() + 1
        unused_idx = cast(int, unused_idx)
        window_tokens = generated_tokens[b].expand(seq_len + 1, seq_len).masked_fill(~mask, unused_idx)

        # Old way, OOM:
        # window_counts = torch.nn.functional.one_hot(window_tokens).sum(dim=1)[...,:-1] # cut off the unused one

        window_counts = torch.zeros(window_tokens.size(0), unused_idx + 1, dtype=torch.long, device=window_tokens.device)
        window_counts.scatter_add_(dim=1, index=window_tokens, src=torch.ones_like(window_tokens))
        window_counts  = window_counts[...,:-1] # cut off the unused vocab item

        per_token_penalty = penalty ** window_counts
        per_token_penalty = per_token_penalty[confirmed_len:]

        logits[b,:,:unused_idx] = torch.where(
            logits[b,:,:unused_idx] > 0,
            logits[b,:,:unused_idx] / per_token_penalty,
            logits[b,:,:unused_idx] * per_token_penalty,
        )

    return logits
