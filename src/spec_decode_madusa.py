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

from src.models import madusa


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


def crop_kv_cache(past_key_values, new_length, best_path, max_accept_len):
    """
    Crop KV cache to a specific sequence length.
    Handles both DynamicCache objects and tuple format.
    """
    if past_key_values is None:
        return None

    accepted_tree_idx = []
    for i in range(max_accept_len):
        node_idx = best_path[i]
        accepted_tree_idx.append(past_key_values+node_idx)

    keep_idx = torch.concat(range(new_length), accepted_tree_idx)
    if hasattr(past_key_values, "select_indices") or hasattr(past_key_values, "select_idx"):
        return past_key_values.select_index(keep_idx)
    else:
        new_past = []
        for layer_past in past_key_values:
            # NGramModel-style cache: a single tensor per layer
            if isinstance(layer_past, torch.Tensor):
                # Crop along the sequence-length dimension (assumed last)
                new_past.append(layer_past[..., :new_length])
            # Standard (key, value) pair cache from HF
            elif len(layer_past) == 2:
                key_state, value_state = layer_past
                k_cropped = key_state[..., :new_length, :]
                v_cropped = value_state[..., :new_length, :]
                new_past.append((k_cropped, v_cropped))
        return tuple(new_past)

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

def default_tree(preset_type:str):
    if preset_type.lower() not in ["lightweight", "greedy_linear", "standard"]:
        raise ValueError("Must be one of the following options: ['lightweight', 'greedy_linear', 'standard']")
    if preset_type.lower() == "lightweight":
        """
        16-node tree focused on high-probability top-1/top-2 branches
        Reduces significantly matrix multiplication sizes and tree-attention mask generation overhead during the verification step.
        Ideal for memory-constrained devices or edge deployment.
        """
        return [
            [0], [0, 0], [0, 0, 0], [0, 0, 0, 0],
            [1], [0, 1], [1, 0], [0, 0, 1],
            [2], [0, 2], [2, 0], [0, 1, 0],
            [3], [0, 0, 2], [1, 1], [0, 0, 0, 1]
        ]
    elif preset_type.lower() == "greedy_linear":
        """
        Minimal single-path execution without branching
        Simplest memory footprint; eliminates complex 2D branching logic and minimizes KV
        cache slicing operations.
        """
        return [
            [0],
            [0, 0],
            [0, 0, 0],
            [0, 0, 0, 0]
        ]
    elif preset_type.lower() == "standard":
        """
        Standard 64-Node Tree
        Maximizes the expected token acceptance per iteration rate; explores a diverse range
        of branches across up to 4 Medusa heads.
        """
        return[
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
    num_heads = logits.size(dim=1)
    top_token_per_head = []
    for i in range(num_heads):
        max_rank = max(tree_choice[i])
        head_logits = logits[0,i,:]
        filter_logits = filter_logprobs(
            nn.LogSoftmax(head_logits, dim=1),
            top_k=top_k,
            top_p=top_p
        )
        top_ids = torch.topk(logits[0, i], k=max_rank + 1).indices
        top_token_per_head.append(top_ids)

    nodes = []
    node_dict = {}
    paths = []

    for path in tree_choice:
        node_path = []
        for depth in range(len(path)):
            rank = path[depth]
            token_id = top_token_per_head[depth][rank]
            prefix = (depth, rank, token_id)

            if prefix not in node_dict:
                new_node_idx = len(nodes)
                node_dict[prefix] = new_node_idx

                parent_prefix = prefix[:-1]
                parent_idx = node_dict[parent_prefix] if len(parent_prefix)>0 else None

                nodes.append(
                    {
                        "node_idx": new_node_idx,
                        "token_id": token_id,
                        "depth": depth,
                        "parent_idx": parent_idx
                    }
                )
                node_path.append(node_dict[parent_prefix])
    size = len(nodes)

    draft_tree_tokens = torch.zeros(1, size)
    pos_idx = torch.zeros(1, size)

    for idx in range(size):
        draft_tree_tokens[0,idx] = nodes[idx]["token_id"]
        pos_idx[0, idx] = cur_gen_idx + nodes[idx]["depth"]

    attn_mask = torch.full(
        size=(size, past_kv_len + size),
        fill_value=-float('inf')
    )
    attn_mask[:, :, :, :past_kv_len] = 0.0

    for i in range(size):
        cur_node = nodes[i]
        while cur_node is not None:
            attn_mask[0, 0, i, past_kv_len+cur_node["node_idx"]] = 0.0
            cur_node = nodes[cur_node["parent_idx"]] if cur_node["parent_idx"] is not None else None

    return {
        "tokens": draft_tree_tokens,
        "attention": attn_mask,
        "pos_idx": pos_idx,
        "paths": paths,
        "nodes": nodes
    }

def speculative_decode(
    target_model,
    tokenizer,
    input_ids,
    mode: Literal["greedy", "sample"],
    medusa: nn.Module | None,
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
        medusa = madusa.madusa(base_model=target_model)

    if isinstance(tree_choices, str):
        tree_choices = default_tree(tree_choices)

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
        return apply_repetition_penalty(logits, ctx, repetition_penalty)

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
            medusa_logits = medusa(last_hidden)

            # Step 1: parallel draft candidate tree generation via the medusa heads
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

            _ = draft_end and draft_end.record()

            total_draft_tokens += draft_tree_tokens.size(-1)

            # Step 2: Target Model Parallel Verification Pass over Candidate Tree
            _ = verifier_start and verifier_start.record()
            target_out = target_model(
                input_ids=draft_tree_tokens,
                past_kv_values=target_kv_cache,
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

            # Step 3: evaluate candidate of tree paths to find the longest valid branch
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
                    node_i = path[depth]
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
                    target_dist = apply_filters(nn.LogSoftmax(node_raw_logits, dim=-1))
                    verified_token = select_index(target_dist)

                    if draft_token == verified_token:
                        accepted_in_path.append(draft_token)
                    else:
                        bonus_token = verified_token
                        bonus_logits = node_raw_logits
                        path_matched = False
                        break

                if path_matched and bonus_token is None:
                    last_node_idx = path[-1]
                    bonnus_raw_logits = penalize_logits(
                        verify_logits[:, last_node_idx, :],
                        confirmed_len=cur_gen_idx + len(path),
                        draft_so_far=generated_tokens,
                    )
                    bonus_token = select_index(
                        apply_filters(
                            nn.LogSoftmax(bonnus_raw_logits, dim=-1)
                        )
                    )
                    bonus_logits = bonnus_raw_logits

                if len(accepted_in_path) > max_accept_len:
                    max_accept_len = len(accepted_in_path)
                    best_accepted_token = accepted_in_path
                    best_bonus_token = bonus_token
                    best_bonus_logits = bonus_logits
                    best_path = path

            # Step 4: updating octile acceptance counters
            gen_offset = cur_gen_idx - prompt_len
            draft_depth = len(best_path) if best_path is not None else 0

            for i in range(len(octile_offsets)):
                checkpoint = octile_offsets[i]
                if gen_offset <= checkpoint and checkpoint < (gen_offset + draft_depth):
                    per_position_draft_count[i] = per_position_draft_count[i] + 1
                    rel_depth = checkpoint - gen_offset
                    if rel_depth == checkpoint - gen_offset:
                        per_position_accept_count[i] = per_position_accept_count[i] + 1

            # Step 5: commit accepted tokens and update seq len
            tokens_to_add = torch.concat((best_accepted_token, best_bonus_token))
            new_gen_idx = cur_gen_idx + tokens_to_add.size(dim=-1)
            generated_tokens[:, cur_gen_idx:new_gen_idx] = tokens_to_add

            total_matched_tokens += len(best_accepted_token)

            # Step 6: prune any unused tree kv cache and extract hidden state for next medusa pass
            target_kv_cache = crop_kv_cache(
                target_out.past_key_values,
                new_gen_idx - 1,
                best_path,
                max_accept_len,
            )

            if max_accept_len > 0:
                last_node_idx = best_path[max_accept_len - 1]
                last_hidden = target_out.hidden_states[-1][:, last_node_idx:last_node_idx + 1, :]

            prev_target_logits = best_bonus_logits
            cur_gen_idx = new_gen_idx

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
