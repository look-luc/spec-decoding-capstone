from typing import Mapping

import torch
from tqdm import tqdm

from src.data.create_inputs import (  # functions to create the prompts and inputs for draft models
    create_inputs,
    create_prompt,
)
from src.models.eagle import EagleModule
from src.spec_decode_eagle import spec_decode_eagle
from src.tasks.translation import compute_eval_metrics, load_data
from src.utils import load_model


def run_divergence_audit(config):
    device = config.device
    model_name = config.target_model if config.target_model is not None else getattr(config, "base_model", None)
    checkpoint_path = "checkpoints/eagle_npi/eagle_module_best.pt"

    print(f"Loading tokenizer and target model: {model_name}...")
    target_model, tokenizer = load_model(str(model_name), device=config.device, draft_model_type=config.draft_model_type)
    target_model.eval()

    print(f"Loading EAGLE module from: {checkpoint_path}...")
    draft_model = EagleModule(
        vocab_size=target_model.config.vocab_size,
        embed_dim=target_model.config.hidden_size,
        hidden_dim=target_model.config.hidden_size,
        num_heads=8
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    draft_model.load_state_dict(state_dict)
    draft_model.eval()

    # Pick a representative test prompt from your dataset
    dataset, language = load_data(config, tokenizer)
    eval_dataset = dataset['test']
    row = eval_dataset.select(range(1))[0]
    assert isinstance(row, Mapping)
    prompt_text = create_prompt(config.task, language, row['source'])
    inputs = create_inputs(prompt_text, tokenizer, device)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_new_tokens = config.max_new_tokens
    repetition_penalty = config.repetition_penalty

    print("\n--- Running Target-Only Generation ---")
    with torch.no_grad():
        target_out = target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    print("--- Running Speculative Decoding ---")
    with torch.no_grad():
        spec_out, metrics = spec_decode_eagle(
            target_model=target_model,
            eagle_module=draft_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            mode=config.decoding_mode,
            max_new_tokens=max_new_tokens,
            tree_choices=config.tree_choices,
            repetition_penalty=repetition_penalty,
            device=device
        )

    target_tokens = target_out[0].tolist()
    spec_tokens = spec_out[0].tolist()

    target_decoded = tokenizer.decode(target_tokens, skip_special_tokens=True)
    spec_decoded = tokenizer.decode(spec_tokens, skip_special_tokens=True)

    print(f"\nTarget Text:\n{target_decoded}")
    print(f"\nSpec Text:\n{spec_decoded}")

    # Find the exact point of divergence
    min_len = min(len(target_tokens), len(spec_tokens))
    first_diff = next((i for i in range(min_len) if target_tokens[i] != spec_tokens[i]), None)

    print("\n" + "="*40)
    if first_diff is None and len(target_tokens) == len(spec_tokens):
        print("✅ Perfect Match! Token sequences are identical.")
    else:
        if first_diff is None:
            first_diff = min_len
        print(f"❌ First Token Divergence at index {first_diff}:")
        if first_diff < len(target_tokens):
            t_tok = target_tokens[first_diff]
            print(f"  Target Token ID {t_tok}: '{tokenizer.decode([t_tok])}'")
        else:
            print("  Target Token ID: [End of Sequence]")

        if first_diff < len(spec_tokens):
            s_tok = spec_tokens[first_diff]
            print(f"  Spec   Token ID {s_tok}: '{tokenizer.decode([s_tok])}'")
        else:
            print("  Spec Token ID: [End of Sequence]")
    print("="*40)
