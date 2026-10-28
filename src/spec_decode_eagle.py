import math
import time
from typing import Literal, cast

import torch
import torch.nn as nn


def get_stop_token_ids(tokenizer, eos_token_id=None):
    stop_ids = set()
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    stop_tokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "</s>",
        "<end_of_turn>",
        "<eos>",
        "[/INST]",
    ]

    for token in stop_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids and len(ids) == 1:
                stop_ids.add(ids[0])
        except Exception:
            pass

    return stop_ids


def sliding_window(tokens, window):
    if window <= 0 or len(tokens) <= window:
        return tokens
    return tokens[-window:]


def extract_token_ids(tree_nodes, device=None):
    ids = []
    for node in tree_nodes:
        tid = node.token_id
        if isinstance(tid, torch.Tensor):
            tid = tid.item()
        ids.append(tid)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def extract_all_leaf_paths(tree_nodes):
    leaf_paths = []
    for node in tree_nodes:
        if len(node.children) == 0:
            path = []
            curr_node = node
            while curr_node is not None:
                path.append(curr_node)
                curr_node = curr_node.parent
            leaf_paths.append(path[::-1])
    return leaf_paths


def crop_kv_cache(past_key_values, new_length):
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        try:
            past_key_values.crop(new_length)
            return past_key_values
        except Exception:
            pass

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            if k is not None and isinstance(k, torch.Tensor):
                past_key_values.key_cache[i] = k[..., :new_length, :]
            if v is not None and isinstance(v, torch.Tensor):
                past_key_values.value_cache[i] = v[..., :new_length, :]
        if hasattr(past_key_values, "_seen_tokens"):
            past_key_values._seen_tokens = new_length
        return past_key_values

    if isinstance(past_key_values, (tuple, list)):
        new_past = []
        for layer_past in past_key_values:
            if isinstance(layer_past, torch.Tensor):
                new_past.append(layer_past[..., :new_length])
            elif isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                key_state, value_state = layer_past
                if isinstance(key_state, torch.Tensor) and isinstance(value_state, torch.Tensor):
                    new_past.append((key_state[..., :new_length, :], value_state[..., :new_length, :]))
                else:
                    new_past.append(layer_past)
            else:
                new_past.append(layer_past)
        return tuple(new_past)

    return past_key_values


def get_kv_cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
        for k in past_key_values.key_cache:
            if k is not None and isinstance(k, torch.Tensor):
                return k.size(2) if len(k.shape) == 4 else k.size(-1)
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        first_item = past_key_values[0]
        if isinstance(first_item, torch.Tensor):
            return first_item.size(-1)
        elif isinstance(first_item, (tuple, list)) and len(first_item) > 0:
            return first_item[0].size(2) if len(first_item[0].shape) == 4 else first_item[0].size(-1)
    return 0


def crop_align_eagle_kv_cache(kv_cache, accepted_path_nodes, base_seq_len):
    if kv_cache is None:
        return None

    accepted_idxs = [node.index for node in accepted_path_nodes]
    path_indices = [base_seq_len + i for i in accepted_idxs]

    if hasattr(kv_cache, "crop") and len(accepted_path_nodes) == 0:
        kv_cache.crop(base_seq_len)
        return kv_cache

    if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
        for layer_idx in range(len(kv_cache.key_cache)):
            k_state = kv_cache.key_cache[layer_idx]
            v_state = kv_cache.value_cache[layer_idx]

            if k_state is not None and v_state is not None and isinstance(k_state, torch.Tensor):
                base_k = k_state[..., :base_seq_len, :]
                base_v = v_state[..., :base_seq_len, :]
                path_k = k_state[..., path_indices, :]
                path_v = v_state[..., path_indices, :]

                kv_cache.key_cache[layer_idx] = torch.cat([base_k, path_k], dim=-2)
                kv_cache.value_cache[layer_idx] = torch.cat([base_v, path_v], dim=-2)

        if hasattr(kv_cache, "_seen_tokens"):
            kv_cache._seen_tokens = base_seq_len + len(accepted_idxs)
        return kv_cache

    if isinstance(kv_cache, (tuple, list)):
        updated_kv_cache = []
        for layer_kv in kv_cache:
            if not isinstance(layer_kv, (tuple, list, torch.Tensor)):
                updated_kv_cache.append(layer_kv)
                continue
            if isinstance(layer_kv, torch.Tensor):
                new_state = torch.cat([layer_kv[..., :base_seq_len], layer_kv[..., path_indices]], dim=-1)
                updated_kv_cache.append(new_state)
            elif len(layer_kv) >= 2:
                key_state, value_states = layer_kv[0], layer_kv[1]
                if isinstance(key_state, torch.Tensor) and isinstance(value_states, torch.Tensor):
                    new_keys = torch.cat([key_state[..., :base_seq_len, :], key_state[..., path_indices, :]], dim=-2)
                    new_values = torch.cat([value_states[..., :base_seq_len, :], value_states[..., path_indices, :]], dim=-2)
                    updated_kv_cache.append((new_keys, new_values))
                else:
                    updated_kv_cache.append(layer_kv)
            else:
                updated_kv_cache.append(layer_kv)
        return tuple(updated_kv_cache)

    return kv_cache


def get_node_depth(node):
    depth = 0
    curr = node
    while curr.parent is not None:
        depth += 1
        curr = curr.parent
    return depth


def get_ancestor_tokens(node):
    tokens = []
    curr_node = node
    while curr_node is not None:
        tokens.append(curr_node.token_id)
        curr_node = curr_node.parent
    return tokens[::-1]


def build_eagle_tree_attn(tree_nodes, base_seq_len, device):
    num_nodes = len(tree_nodes)
    total_len = base_seq_len + num_nodes

    attn_mask = torch.full((num_nodes, total_len), -10000.0, dtype=torch.float32, device=device)
    pos_ids = torch.zeros(num_nodes, dtype=torch.long, device=device)

    for i in range(num_nodes):
        node_i = tree_nodes[i]
        depth = node_i.depth if hasattr(node_i, "depth") else get_node_depth(node=node_i)

        pos_ids[i] = base_seq_len + depth
        attn_mask[i, :base_seq_len] = 0.0

        cur_ancestor = node_i
        while cur_ancestor is not None:
            attn_mask[i, base_seq_len + cur_ancestor.index] = 0.0
            cur_ancestor = cur_ancestor.parent

    return attn_mask.unsqueeze(0).unsqueeze(0), pos_ids.unsqueeze(0)


class Node:
    def __init__(self, index, parent, token_id, hidden_state, depth, log_score):
        self.index = index
        self.parent = parent
        self.token_id = token_id
        self.hidden_state = hidden_state
        self.depth = depth
        self.log_score = log_score
        self.children = []

        if parent is not None:
            parent.children.append(self)


def create_node(index, parent, token_id, hidden_state, depth, log_score):
    return Node(index, parent, token_id, hidden_state, depth, log_score)


def eagle_draft_tree_expansion(
    eagle_module,
    root_hidden,
    root_token_id,
    tree_choice: list[int],
    top_k,
    top_p,
    repetition_penalty,
    repetition_penalty_window,
    generated_tokens,
    cur_gen_idx,
    mode,
    score_threshold: float = 0.005
):
    tree_nodes = []
    log_threshold = math.log(max(score_threshold, 1e-12))
    device = root_hidden.device

    if root_hidden.ndim > 1:
        root_hidden = root_hidden.squeeze(0)
    if root_token_id.ndim > 0:
        root_token_id = root_token_id.squeeze()

    root_node = create_node(
        index=0,
        parent=None,
        token_id=root_token_id,
        hidden_state=root_hidden,
        depth=0,
        log_score=0.0
    )

    tree_nodes.append(root_node)
    active_parents = [root_node]

    for depth in range(len(tree_choice)):
        branch_factor = tree_choice[depth]
        if len(active_parents) == 0:
            break

        batch_path_tokens = []
        batch_path_hiddens = []

        for parent in active_parents:
            path_nodes = []
            curr = parent
            while curr is not None:
                path_nodes.append(curr)
                curr = curr.parent
            path_nodes.reverse()

            p_tokens = torch.stack([node.token_id for node in path_nodes]).view(-1)
            p_hiddens = torch.stack([node.hidden_state for node in path_nodes])
            batch_path_tokens.append(p_tokens)
            batch_path_hiddens.append(p_hiddens)

        path_tokens_tensor = torch.stack(batch_path_tokens, dim=0).to(device)
        path_hiddens_tensor = torch.stack(batch_path_hiddens, dim=0).to(device)

        token_embs = eagle_module.get_embedding(path_tokens_tensor)
        fused_feat = torch.cat([token_embs, path_hiddens_tensor], dim=-1)
        projected = eagle_module.fc_fusion(fused_feat)

        seq_len = projected.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device) if seq_len > 1 else None

        hidden = projected
        for layer in eagle_module.decoder_layers:
            hidden = layer(
                hidden,
                hidden,
                tgt_mask=tgt_mask,
                tgt_is_csal=(seq_len > 1)
            )

        next_hiddens = hidden[:, -1, :]  # Shape: (num_parents, hidden_dim)
        draft_raw_logits = eagle_module.lm_head(next_hiddens)  # Shape: (num_parents, vocab_size)

        processed_logprobs_list = []
        for i, parent in enumerate(active_parents):
            logits_i = draft_raw_logits[i : i + 1]  # (1, vocab_size)

            if repetition_penalty != 1.0:
                ancestor_token_list = get_ancestor_tokens(parent)
                ancestor_tensor = torch.tensor(ancestor_token_list, dtype=generated_tokens.dtype, device=device)
                full_context = torch.cat(
                    [generated_tokens[0, :cur_gen_idx], ancestor_tensor],
                    dim=-1
                )
                penalty_context = sliding_window(full_context, window=repetition_penalty_window)
                logits_i = apply_repetition_penalty(logits_i, penalty_context, repetition_penalty)

            logprobs_i = filter_logprobs(
                torch.log_softmax(logits_i, dim=-1),
                top_k=top_k,
                top_p=top_p
            )
            processed_logprobs_list.append(logprobs_i.squeeze(0))

        processed_logprobs = torch.stack(processed_logprobs_list, dim=0)

        top_scores, top_token_ids = torch.topk(processed_logprobs, k=branch_factor, dim=-1)

        next_parents = []
        for i, parent in enumerate(active_parents):
            for k_idx in range(branch_factor):
                child_log_score = parent.log_score + top_scores[i, k_idx].item()

                if child_log_score >= log_threshold:
                    child_node = create_node(
                        index=len(tree_nodes),
                        parent=parent,
                        token_id=top_token_ids[i, k_idx],
                        hidden_state=next_hiddens[i],
                        depth=depth + 1,
                        log_score=child_log_score
                    )
                    tree_nodes.append(child_node)
                    next_parents.append(child_node)

        active_parents = next_parents

    return tree_nodes


def spec_decode_eagle(
    target_model,
    eagle_module,
    tokenizer,
    input_ids: torch.Tensor,
    mode: Literal["greedy", "sample"],
    max_new_tokens: int = 128,
    tree_choices: str | list[int] = [1, 4, 2, 3],
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    repetition_penalty_window: int = 16,
    score_threshold: float = 0.01,
    eos_token_id: int | None = None,
    device=None,
    track_iterations: bool = False
):
    bs = input_ids.size(0)
    assert bs == 1, "Speculative decoding only supports batch_size=1"

    if device is None:
        device = next(target_model.parameters()).device

    if isinstance(tree_choices, str):
        tc_lower = tree_choices.lower()
        if tc_lower in ["eagle_1", "eagle-1", "eagle 1"]:
            tree_choices_arr = [1, 4, 2, 2]
        elif tc_lower in ["eagle_2", "eagle-2", "eagle 2"]:
            tree_choices_arr = [1, 8, 4, 2]
        elif tc_lower == "custom":
            tree_choices_arr = [1, 4, 2, 3]
        elif tc_lower == "conservative":
            tree_choices_arr = [1, 3, 2, 1]
        elif tc_lower in ["balance_2", "balance-2", "balance 2"]:
            tree_choices_arr = [1, 5, 2, 1]
        elif tc_lower in ["balance_1", "balance-1", "balance 1"]:
            tree_choices_arr = [1, 4, 2, 1]
        elif tc_lower in ["shallow 1", "shallow_1", "shallow-1", "shallow1"]:
            tree_choices_arr = [1, 8, 2]
        elif tc_lower in ["shallow 2", "shallow_2", "shallow-2", "shallow2"]:
            tree_choices_arr = [1, 12]
        else:
            raise ValueError(f"{tree_choices} is not a supported tree option.")
    else:
        tree_choices_arr = tree_choices

    def apply_filters(logprobs: torch.Tensor) -> torch.Tensor:
        return filter_logprobs(logprobs, top_k=top_k, top_p=top_p)

    def select_index(logprobs: torch.Tensor):
        return sample(logprobs, mode)

    def penalize_logits(logits: torch.Tensor, confirmed_len: int) -> torch.Tensor:
        if repetition_penalty == 1.0:
            return logits
        ctx = generated_tokens[:, max(0, confirmed_len - repetition_penalty_window):confirmed_len]
        return apply_repetition_penalty(logits, ctx, repetition_penalty)

    stop_token_ids = torch.tensor(list(get_stop_token_ids(tokenizer, eos_token_id)), device=device)
    input_ids = input_ids.to(device)

    generated_tokens = torch.cat(
        [input_ids, torch.zeros(input_ids.size(0), max_new_tokens, device=device, dtype=torch.int64)],
        dim=-1,
    )

    prompt_len = input_ids.size(dim=-1)
    cur_gen_idx = prompt_len

    total_draft_tokens = 0
    total_matched_tokens = 0
    num_iterations = 0

    max_depth = len(tree_choices_arr)
    per_position_draft_count = [0] * max_depth
    per_position_accept_count = [0] * max_depth
    octile_offsets = list(range(1, max_depth + 1))

    draft_times_acc = [0.0, 0.0, 0]
    verifier_times_acc = [0.0, 0.0, 0]

    def get_time():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()

    with torch.no_grad():
        start_time = get_time()

        target_out = target_model(input_ids, use_cache=True, output_hidden_states=True)
        target_kv_cache = target_out.past_key_values

        last_hidden = target_out.hidden_states[-1][:, -1, :]
        first_logits = penalize_logits(target_out.logits[:, -1, :], confirmed_len=cur_gen_idx)
        first_logprobs = apply_filters(torch.log_softmax(first_logits, dim=-1))
        first_token = select_index(first_logprobs)

        generated_tokens[:, cur_gen_idx] = first_token
        cur_gen_idx += 1

        while cur_gen_idx < generated_tokens.size(-1):
            num_iterations += 1
            curr_token = generated_tokens[:, cur_gen_idx - 1]

            t_d0 = get_time()
            tree_nodes = eagle_draft_tree_expansion(
                eagle_module=eagle_module,
                root_hidden=last_hidden,
                root_token_id=curr_token,
                tree_choice=tree_choices_arr,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_penalty_window=repetition_penalty_window,
                generated_tokens=generated_tokens,
                cur_gen_idx=cur_gen_idx,
                mode=mode,
                score_threshold=score_threshold
            )
            t_d1 = get_time()

            d_ms = (t_d1 - t_d0) * 1000.0
            draft_times_acc[0] += d_ms
            draft_times_acc[1] += d_ms ** 2
            draft_times_acc[2] += 1

            for node in tree_nodes[1:]:
                if 1 <= node.depth <= max_depth:
                    per_position_draft_count[node.depth - 1] += 1

            flattened_tokens = extract_token_ids(tree_nodes=tree_nodes, device=device)
            base_seq_len = get_kv_cache_length(target_kv_cache)

            tree_attn_mask_list, tree_pos_ids_list = build_eagle_tree_attn(
                tree_nodes=tree_nodes,
                base_seq_len=base_seq_len,
                device=device
            )
            model_dtype = next(target_model.parameters()).dtype

            tree_pos_ids = tree_pos_ids_list.detach().clone().to(dtype=torch.long, device=device)
            tree_attn_mask = tree_attn_mask_list.detach().clone().to(dtype=model_dtype, device=device)

            t_v0 = get_time()
            target_out = target_model(
                input_ids=flattened_tokens,
                attention_mask=tree_attn_mask,
                position_ids=tree_pos_ids,
                past_key_values=target_kv_cache,
                use_cache=True,
                output_hidden_states=True
            )
            t_v1 = get_time()

            v_ms = (t_v1 - t_v0) * 1000.0
            verifier_times_acc[0] += v_ms
            verifier_times_acc[1] += v_ms ** 2
            verifier_times_acc[2] += 1

            target_raw_logits = target_out.logits
            verifier_hiddens = target_out.hidden_states[-1]

            penalize_target_logits = apply_repetition_penalty_batched(
                logits=target_raw_logits,
                generated_tokens=torch.cat([generated_tokens[:, :cur_gen_idx], flattened_tokens], dim=-1),
                confirmed_len=cur_gen_idx,
                penalty=repetition_penalty,
                window=repetition_penalty_window
            )
            target_logprobs = filter_logprobs(
                torch.log_softmax(penalize_target_logits, dim=-1),
                top_k=top_k,
                top_p=top_p
            )

            candidate_paths = extract_all_leaf_paths(tree_nodes=tree_nodes)
            best_path = []
            best_accept_count = -1
            bonus_token = None
            new_last_hidden = None

            for path in candidate_paths:
                accepted_in_paths = []
                for step in range(len(path) - 1):
                    parent_node = path[step]
                    child_node = path[step + 1]

                    pred_token = select_index(target_logprobs[:, parent_node.index, :])

                    if pred_token == child_node.token_id:
                        accepted_in_paths.append(child_node)
                    else:
                        break

                if len(accepted_in_paths) > best_accept_count:
                    best_accept_count = len(accepted_in_paths)
                    best_path = accepted_in_paths

                    last_accepted_node = accepted_in_paths[-1] if len(accepted_in_paths) else path[0]
                    bonus_token = select_index(target_logprobs[:, last_accepted_node.index, :])
                    new_last_hidden = verifier_hiddens[:, last_accepted_node.index, :]

            for node in best_path:
                if 1 <= node.depth <= max_depth:
                    per_position_accept_count[node.depth - 1] += 1

            accepted_tokens = [node.token_id for node in best_path]
            accepted_tensor = torch.tensor(accepted_tokens, dtype=torch.int64, device=device).view(-1)
            bonus_tensor = bonus_token.view(-1)

            tokens_to_add = torch.cat([accepted_tensor, bonus_tensor], dim=0).unsqueeze(0)
            num_added = tokens_to_add.size(-1)

            new_gen_idx = cur_gen_idx + num_added

            if new_gen_idx > generated_tokens.size(-1):
                padding = torch.zeros(
                    (generated_tokens.size(0), new_gen_idx - generated_tokens.size(-1)),
                    dtype=generated_tokens.dtype,
                    device=device
                )
                generated_tokens = torch.cat([generated_tokens, padding], dim=-1)

            generated_tokens[:, cur_gen_idx:new_gen_idx] = tokens_to_add
            cur_gen_idx = new_gen_idx

            total_matched_tokens += best_accept_count
            total_draft_tokens += len(tree_nodes) - 1

            target_kv_cache = crop_align_eagle_kv_cache(
                target_out.past_key_values,
                best_path,
                base_seq_len
            )
            last_hidden = new_last_hidden

            if torch.isin(tokens_to_add, stop_token_ids).any():
                generated_tokens = generated_tokens[:, :cur_gen_idx]
                break

    total_time = get_time() - start_time
    acceptance_rate = total_matched_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    total_generated_tokens = cur_gen_idx - prompt_len

    octile_position_acceptance = [
        acc / draf if draf > 0 else None
        for acc, draf in zip(per_position_accept_count, per_position_draft_count)
    ]

    metrics = {
        "time": total_time,
        "generated_tokens": total_generated_tokens,
        "draft_tokens": total_draft_tokens,
        "matched_tokens": total_matched_tokens,
        "acceptance_rate": acceptance_rate,
        "octile_position_acceptance": octile_position_acceptance,
        "octile_positions": octile_offsets,
        "num_iterations": num_iterations,
        "toks_per_sec": total_generated_tokens / total_time if total_time > 0 else 0,
    }

    if draft_times_acc[2] > 0 and verifier_times_acc[2] > 0:
        d_sum, d_sum_sq, d_n = draft_times_acc
        v_sum, v_sum_sq, v_n = verifier_times_acc
        average_draft_time = d_sum / d_n
        average_verifier_time = v_sum / v_n
        metrics["average_draft_time"] = average_draft_time / 1000.0
        metrics["average_verifier_time"] = average_verifier_time / 1000.0
        metrics["draft_time_variance"] = max(d_sum_sq / d_n - average_draft_time**2, 0.0) / 1e6
        metrics["verifier_time_variance"] = max(v_sum_sq / v_n - average_verifier_time**2, 0.0) / 1e6

    return generated_tokens, metrics


def filter_logprobs(logprobs: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    filtered = logprobs
    if top_k > 0:
        filtered = apply_top_k(filtered, k=top_k)
    if 0.0 < top_p < 1.0:
        filtered = apply_top_p(filtered, p=top_p)
    if top_k > 0 or 0.0 < top_p < 1.0:
        filtered = torch.log_softmax(filtered, dim=-1)
    return filtered


def sample(logprobs: torch.Tensor, mode: Literal["greedy", "sample"]):
    if mode == "greedy":
        return logprobs.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logprobs).sample()


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k < 1:
        raise ValueError(f"top_k must be >= 1, got {k}")
    if k >= logits.size(-1):
        return logits
    top_values, _ = torch.topk(logits, k, dim=-1)
    kth_value = top_values[..., -1, None]
    indices_to_remove = logits < kth_value
    return logits.masked_fill(indices_to_remove, float("-inf"))


def top_k_sampling(logprobs, k):
    if logprobs.ndim > 1:
        logprobs = logprobs.squeeze(0)
    sorted_logprobs, sorted_idxs = torch.sort(logprobs, descending=True, dim=-1)
    return sorted_logprobs[:k], sorted_idxs[:k]


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"top_p must be between 0.0 and 1.0, got {p}")
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def apply_repetition_penalty(logits: torch.Tensor, context_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or context_ids.size(-1) == 0:
        return logits
    for b in range(logits.size(0)):
        max_vocab_index = cast(int, torch.max(context_ids[b]).item() + 1)
        window_counts = torch.zeros(max_vocab_index, dtype=torch.long, device=context_ids[b].device)
        window_counts.scatter_add_(dim=0, index=context_ids[b], src=torch.ones_like(context_ids[b]))
        per_token_penalty = penalty ** window_counts
        logits[b, :max_vocab_index] = torch.where(
            logits[b, :max_vocab_index] > 0,
            logits[b, :max_vocab_index] / per_token_penalty,
            logits[b, :max_vocab_index] * per_token_penalty,
        )
    return logits


def apply_repetition_penalty_batched(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    confirmed_len: int,
    penalty: float,
    window: int,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    bs, seq_len = generated_tokens.shape
    device = generated_tokens.device

    for b in range(bs):
        mask = ~torch.triu(torch.ones(seq_len + 1, seq_len, dtype=torch.bool, device=device))
        window_start = torch.clamp(torch.arange(seq_len + 1, device=device) - window, 0).unsqueeze(-1)
        mask *= (torch.arange(seq_len, device=device) >= window_start)

        unused_idx = cast(int, torch.max(generated_tokens).item() + 1)
        window_tokens = generated_tokens[b].expand(seq_len + 1, seq_len).masked_fill(~mask, unused_idx)

        window_counts = torch.zeros(window_tokens.size(0), unused_idx + 1, dtype=torch.long, device=device)
        window_counts.scatter_add_(dim=1, index=window_tokens, src=torch.ones_like(window_tokens))
        window_counts = window_counts[..., :-1]

        per_token_penalty = penalty ** window_counts
        num_verify_tokens = logits.size(1)
        per_token_penalty = per_token_penalty[confirmed_len : confirmed_len + num_verify_tokens]

        logits[b, :, :unused_idx] = torch.where(
            logits[b, :, :unused_idx] > 0,
            logits[b, :, :unused_idx] / per_token_penalty,
            logits[b, :, :unused_idx] * per_token_penalty,
        )
    return logits
