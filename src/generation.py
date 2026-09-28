import os
import time
from typing import cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config.config import ExperimentConfig
from src.models.eagle import EagleModule
from src.models.madusa import madusa
from src.n_gram import NGramModel
from src.spec_decode import speculative_decode
from src.spec_decode_eagle import spec_decode_eagle
from src.spec_decode_madusa import speculative_decode as madusa_spec


def generate_output(
    inputs: dict,
    model,
    tokenizer: PreTrainedTokenizer,
    draft_model: PreTrainedModel | NGramModel | None,
    draft_tokenizer: PreTrainedTokenizer | None,
    config: ExperimentConfig,
) -> tuple[str, dict]:
    """Generates an output, optionally using speculative decoding."""
    same_tokenizer = (draft_tokenizer is None) or (
        tokenizer.vocab_size == draft_tokenizer.vocab_size
    )
    is_cuda = inputs["input_ids"].device.type == "cuda"
    prompt_len = inputs["input_ids"].shape[1]

    def get_time():
        if is_cuda:
            torch.cuda.synchronize()
        return time.time()

    # Use our custom spec dec implementation
    if config.draft_model_type != "none" and not config.use_hf_assisted:
        if not same_tokenizer:
            raise NotImplementedError("SD with different tokenizers not implemented.")
        output_ids, metrics = speculative_decode(
            target_model=model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            mode=config.decoding_mode,
            max_new_tokens=config.max_new_tokens,
            gamma=config.gamma,  # type:ignore
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            repetition_penalty_window=config.repetition_penalty_window,
            device=inputs["input_ids"].device,
            track_iterations=config.track_iterations,
        )
        decoded = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        decoded = cast(str, decoded).strip()
        return decoded, metrics
    if isinstance(draft_model, madusa) or config.draft_model_type.lower() == "medusa":
        num_heads = config.num_heads
        output_ids, metrics = madusa_spec(
            target_model=model,
            medusa=draft_model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            mode=config.decoding_mode,
            num_heads=num_heads,
            max_new_tokens=config.max_new_tokens,
            tree_choices=getattr(config, "tree_choices", "greedy_linear"),
            device=inputs["input_ids"].device,
        )
        decoded = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        return cast(str, decoded).strip(), metrics
    if isinstance(draft_model, EagleModule) or config.draft_model_type.lower() == "eagle":
        eagle_module = EagleModule(
            vocab_size=model.config.vocab_size,
            embed_dim=model.config.hidden_size,
            hidden_dim=model.config.hidden_size,
            num_heads=config.num_heads,
        ).to(config.device)

        eagle_state_dict = torch.load(os.path.join(config.draft_model_dir, "eagle_module.pt"), map_location=config.device)
        eagle_module.load_state_dict(eagle_state_dict)

        output_ids, metrics = spec_decode_eagle(
            target_model=model,
            eagle_module=eagle_module,
            tokenizer=tokenizer,
            input_ids=inputs,
            mode=config.decoding_mode,
            max_new_tokens=config.max_new_tokens,
            tree_choices=config.tree_choices,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            device=config.device
        )
        decoded = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        return cast(str, decoded).strip(), metrics
    if isinstance(draft_model, NGramModel):
        raise ValueError(
            "NGramModel can only be used with bespoke decoding implementation!"
        )

    # Otherwise, use HF decoding
    with torch.no_grad():
        # Measure prefill time (one forward pass to fill KV cache)
        prefill_start = get_time()
        model(inputs["input_ids"], use_cache=True)
        prefill_time = get_time() - prefill_start

        if config.draft_model_type != "none":
            generate_kwargs = {
                "assistant_model": draft_model,
                "num_assistant_tokens": config.gamma,
                "num_assistant_tokens_schedule": config.hf_schedule,
            }
            if not same_tokenizer:
                generate_kwargs["tokenizer"] = tokenizer
                generate_kwargs["assistant_tokenizer"] = draft_tokenizer
        else:
            generate_kwargs = {}

        # Generate (this re-does prefill internally)
        gen_start = get_time()
        out = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
        total_time = get_time() - gen_start

    decode_time = total_time - prefill_time

    # Decode only the new tokens (after the prompt)
    generated_token_count = out.shape[1] - prompt_len
    decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    decoded = cast(str, decoded).strip()
    return decoded, {
        "generated_tokens": generated_token_count,
        "time": decode_time,
        "toks_per_sec": generated_token_count / decode_time if decode_time > 0 else 0,
    }
