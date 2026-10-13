import logging
import os
import time
from dataclasses import asdict
from typing import Any

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.dataset import assemble_dataset

logger = logging.getLogger(__name__)


def compute_eagle_loss(
    eagle_module: nn.Module,
    base_model: nn.Module,
    batch: dict,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = next(base_model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_logprobs = batch["topk_logprobs"].to(device)
    target_indices = batch["topk_logprobs_indices"].to(device)
    label_mask = batch["label_mask"].to(device)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        with torch.no_grad():
            base_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            base_hidden_states = base_outputs.hidden_states[-1]

        pred_hidden, pred_logits = eagle_module(
            token_id=input_ids[:, :-1],
            hidden_state=base_hidden_states[:, :-1, :],
        )

        shifted_target_logprobs = target_logprobs[:, 1:, :]
        matched_target_indices = target_indices[:, 1:, :]
        mask = label_mask[:, 1:]

        student_logprobs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        gathered_student_logprobs = student_logprobs.gather(
            dim=-1, index=matched_target_indices
        )

        cross_entropy_loss = -(
            torch.exp(shifted_target_logprobs) * gathered_student_logprobs
        ).sum(-1)

        final_loss = (cross_entropy_loss * mask).sum() / mask.sum().clamp(min=1)

    return final_loss


def save_eagle_checkpoint(eagle_module: nn.Module, output_dir: str, label: str = "final"):
    """Saves EAGLE module state dictionary to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"eagle_module_{label}.pt")

    if hasattr(eagle_module, "module"):
        state_dict = eagle_module.module.state_dict()
    else:
        state_dict = eagle_module.state_dict()

    torch.save(state_dict, save_path)
    logger.info(f"Saved EAGLE weights to: {save_path}")

def grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def run_eagle_training(config, base_model, eagle_module, tokenizer):
    device = next(eagle_module.parameters()).device
    os.makedirs(config.output_dir, exist_ok=True)

    if wandb.run is None:
        wandb.init(
            project=getattr(config, "wandb_project", "speculative decoding capstone"),
            config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)
        )

    optimizer = optim.AdamW(eagle_module.parameters(), lr=config.learning_rate)

    lang_code = getattr(config, "language_code", "en")
    dataset_type = getattr(config, "dataset_type", "bi")
    max_samples = max(getattr(config, "max_samples", 7000), 6000)

    logger.info(f"Assembling dataset in memory for '{lang_code}'...")
    splits = assemble_dataset(
        lang_code=lang_code,
        type=dataset_type,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )

    dataset = splits["train"]
    if len(dataset) == 0:
        raise ValueError("The training dataset is empty!")

    def collate_fn(batch):
        bs = len(batch)
        token_lists = []
        for r in batch:
            if "token_ids" in r and r["token_ids"] is not None:
                ids = r["token_ids"]
            elif "text" in r and r["text"] is not None:
                ids = tokenizer.encode(r["text"], add_special_tokens=True)
            elif config.language_code in r and r[config.language_code] is not None:
                ids = tokenizer.encode(r[config.language_code], add_special_tokens=True)
            else:
                string_val = next((v for k, v in r.items() if k != "origin" and isinstance(v, str)), None)
                ids = tokenizer.encode(string_val, add_special_tokens=True)
            token_lists.append(ids)

        seq_len = max(len(ids) for ids in token_lists)
        topk = getattr(config, "top_k", 50) if getattr(config, "top_k", 50) > 0 else 50

        input_ids = torch.full((bs, seq_len), tokenizer.pad_token_id or 0, dtype=torch.long)
        attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
        topk_logprobs = torch.zeros((bs, seq_len - 1, topk), dtype=torch.float32)
        topk_logprobs_indices = torch.zeros((bs, seq_len - 1, topk), dtype=torch.long)
        label_mask = torch.zeros((bs, seq_len - 1), dtype=torch.float32)

        for idx in range(bs):
            ids = token_lists[idx]
            item_seq_len = len(ids)
            item_prompt_len = batch[idx].get("prompt_length", 1)

            input_ids[idx][:item_seq_len] = torch.as_tensor(ids)
            attention_mask[idx][:item_seq_len] = 1

            # If dataset lacks precomputed logprobs, fallback to uniform/dummy or standard CE targets
            if "logprobs" in batch[idx] and batch[idx]["logprobs"] is not None:
                lp = batch[idx]["logprobs"]
                lpi = batch[idx].get("logprobs_vocab_idx", batch[idx].get("topk_logprobs_indices"))
                if lp and lpi:
                    topk_logprobs[idx][item_prompt_len - 1 : item_seq_len - 1] = torch.as_tensor(lp[:item_seq_len - item_prompt_len])
                    topk_logprobs_indices[idx][item_prompt_len - 1 : item_seq_len - 1] = torch.as_tensor(lpi[:item_seq_len - item_prompt_len])

            label_mask[idx][item_prompt_len - 1 : item_seq_len - 1] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "topk_logprobs": topk_logprobs,
            "topk_logprobs_indices": topk_logprobs_indices,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=getattr(config, "batch_size", 16),
        shuffle=True,
        collate_fn=collate_fn,
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    eagle_module.train()

    scheduler = _build_scheduler(optimizer, config)
    start_step = _restore_training_state(config, optimizer, scheduler, device)
    global_step = 0
    max_steps = getattr(config, "max_steps", 1000)
    log_every = getattr(config, "log_every", 10)
    step = start_step
    target_step = start_step + config.max_steps
    accum_count = 0
    log_accum_loss = 0.0
    log_micro_count = 0
    best_eval_loss = float("inf")
    start_time = time.time()

    while global_step < max_steps:
        for batch_idx, batch in enumerate(dataloader):
            if global_step>= max_steps:
                break
            optimizer.zero_grad()
            loss = compute_eagle_loss(eagle_module, base_model, batch, device=device)

            if torch.isnan(loss) or loss.item() == 0.0:
                logger.warning(f"Step {global_step} | Encountered zero or NaN loss. Check target logprobs generation.")

            scaler.scale(loss / config.grad_accum_steps).backward()
            accum_count += 1
            log_accum_loss += loss.item()
            log_micro_count += 1

            if accum_count < config.grad_accum_steps:
                continue

            scaler.unscale_(optimizer)
            unclipped_grad_norm = grad_norm(eagle_module)
            torch.nn.utils.clip_grad_norm_(eagle_module.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Log to Wandb
            if global_step % log_every == 0:
                avg_loss = log_accum_loss / log_micro_count
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
                )
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/grad_norm": unclipped_grad_norm,
                    "train/epoch": global_step,
                    "train/global_step": global_step
                })

    save_eagle_checkpoint(eagle_module, config.output_dir, label="final")

def _restore_training_state(config: Any, optimizer: optim.Optimizer, scheduler: LambdaLR, device: torch.device) -> int:
    start_step = 0
    resume_from = getattr(config, "resume_from", None)
    if resume_from:
        checkpoint_name = os.path.basename(resume_from)
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])

        opt_path = os.path.join(resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            logger.info(f"Loading optimizer state from {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        else:
            logger.warning("No optimizer state found — learning rates will reset")

        sched_path = os.path.join(resume_from, "scheduler.pt")
        if os.path.exists(sched_path):
            logger.info(f"Loading scheduler state from {sched_path}")
            scheduler.load_state_dict(torch.load(sched_path, map_location=device))
        elif start_step > 0:
            logger.warning(
                f"No scheduler state found — fast-forwarding scheduler to step {start_step}"
            )
            for _ in range(start_step):
                scheduler.step()
    return start_step

def _build_scheduler(optimizer: optim.Optimizer, config: Any) -> LambdaLR:
    max_steps = getattr(config, "max_steps", 225)
    warmup_steps = getattr(config, "warmup_steps", 20)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.2, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
