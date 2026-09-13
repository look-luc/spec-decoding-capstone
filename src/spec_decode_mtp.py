import time
from typing import Literal, cast

import torch


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

def crop_kv_cache(past_key_values, new_length):
    """
    Crop KV cache to a specific sequence length.
    Handles both DynamicCache objects and tuple format.
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        past_key_values.crop(new_length)
        return past_key_values
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

def spec_decode_mtp(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    mode: Literal["greedy", "sample"],
    max_new_tokens: int = 128,
    gamma: int = 5,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.1,
    repetition_penalty_window: int = 16,
    eos_token_id: int | None = None,
    device=None,
    track_iterations: bool = False,
):
    """
    Speculative Decoding with KV Caching.
    Key features:

    Args:
        target_model: The large target model
        draft_model: The smaller draft model
        tokenizer: Shared tokenizer (must be same for both models)
        input_ids: Input token IDs [1, seq_len]
        mode: 'greedy' | 'sample'
        max_new_tokens: Maximum new tokens to generate
        gamma: Number of draft tokens to generate per iteration
        top_k: If > 0, only sample from the top k tokens
        top_p: If > 0 and < 1, keep the smallest set of tokens whose cumulative prob >= p
        eos_token_id: End of sequence token ID
        device: Device to run on

    Returns:
        output_ids: Generated token IDs
        metrics: Dict with acceptance_rate, time, draft_tokens, matched_tokens, etc.
    """
    bs = input_ids.size(0)
    assert bs == 1, "Speculative decoding only supports batch_size=1"

    if device is None:
        device = next(target_model.parameters()).device

    def apply_filters(logprobs: torch.Tensor) -> torch.Tensor:
        return filter_logprobs(logprobs, top_k=top_k, top_p=top_p)

    def select_index(logprobs: torch.Tensor):
        return sample(logprobs, mode)

    def penalize_logits(
        logits: torch.Tensor,
        confirmed_len: int,
        draft_so_far: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply windowed repetition penalty to raw logits (single position).

        Builds the penalty context as:
            last `repetition_penalty_window` tokens of
            generated_tokens[:, :confirmed_len] ++ draft_so_far (if any)

        This is called with raw logits (before log_softmax) so the
        positive/negative sign distinction in the penalty formula is meaningful.

        Args:
            logits:       Raw model logits, shape [bs, d_vocab].
            confirmed_len: Number of confirmed tokens in generated_tokens
                          (i.e. cur_gen_idx at the time of the call).
            draft_so_far: Draft tokens generated in the current iteration
                          so far, shape [bs, n_draft]. None or empty = no drafts yet.
        Returns:
            Penalized logits, same shape as input.
        """
        if repetition_penalty == 1.0:
            return logits
        # Build context: confirmed portion of generated_tokens + any draft tokens
        ctx = generated_tokens[:, :confirmed_len]
        if draft_so_far is not None and draft_so_far.size(-1) > 0:
            ctx = torch.cat([ctx, draft_so_far], dim=-1)
        # Slide to the last `repetition_penalty_window` tokens
        ctx = ctx[:, -repetition_penalty_window:]
        return apply_repetition_penalty(logits, ctx, repetition_penalty)

    stop_token_ids = torch.tensor(
        list(get_stop_token_ids(tokenizer, eos_token_id)), device=device
    )
    input_ids = input_ids.to(device)

    # This is okay because if we've gotten this far, we know the actual tokenizers are the same length.
    # Just be aware that logits may have a slightly shorter dimension
    d_vocab = max(draft_model.config.vocab_size, target_model.config.vocab_size)

    # B,S+max_new
    generated_tokens = torch.concat(
        [
            input_ids,
            torch.zeros(
                input_ids.size(0), max_new_tokens, device=device, dtype=torch.int64
            ),
        ],
        dim=-1,
    )
    prompt_len = input_ids.size(-1)
    cur_gen_idx = input_ids.size(-1)

    # Track average time for draft and verifier forward pass for speedup factor
    # Each accumulator: (sum_of_times, sum_of_squared_times, count)
    draft_start,draft_end,verifier_start, verifier_end   = None, None, None, None
    draft_times_acc = (0., 0., 0)
    verifier_times_acc = (0., 0., 0)
    if device.type == 'cuda':
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
        target_out = target_model(input_ids, use_cache=True)
        target_kv_cache = target_out.past_key_values
        draft_kv_cache = draft_model(input_ids, use_cache=True).past_key_values

        # Add the first new token.
        # Penalty context: the last W tokens of the prompt (no generated tokens yet).
        first_logits = penalize_logits(target_out.logits[:, -1, :], confirmed_len=cur_gen_idx)
        last_target_token = select_index(
            apply_filters(torch.log_softmax(first_logits, dim=-1))
        )
        generated_tokens[:, cur_gen_idx] = last_target_token
        cur_gen_idx += 1

        # Metrics
        total_draft_tokens = 0
        total_matched_tokens = 0
        # Per-position acceptance for the octiles (eg 16, 32, ..., 128 if we use max_tokens=128)
        # These are offsets after the prompt length, not absolute indices
        octile_offsets = [i * (max_new_tokens // 8) + 1 for i in range(8)]
        per_position_accept_count = [0] * 8
        per_position_draft_count = [0] * 8
        num_iterations = 0
        iteration_history = []
        start_time = get_time()

        while cur_gen_idx < generated_tokens.size(-1):
            num_iterations += 1
            # Step 1: Draft tokens
            # B * gamma (unless gamma > remaining tokens)
            max_draft_tokens = min(gamma, generated_tokens.size(-1) - cur_gen_idx)

            mtp_out = draft_model.

def apply_repetition_penalty(
    logits: torch.Tensor,
    context_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply multiplicative repetition penalty to raw logits (single position)."""
    if penalty == 1.0 or context_ids.size(-1) == 0:
        return logits

    for b in range(logits.size(0)):
        # window_counts = torch.nn.functional.one_hot(context_ids[b]).sum(dim=-2)
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
