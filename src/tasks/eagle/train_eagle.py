import logging
import os

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

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
        mask = label_mask[:, 1:, :]

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


def run_eagle_training(config, base_model, eagle_module):
    """Execution loop for EAGLE training/distillation."""
    device = next(eagle_module.parameters()).device
    os.makedirs(config.output_dir, exist_ok=True)

    optimizer = optim.AdamW(eagle_module.parameters(), lr=config.learning_rate)

    dataset = datasets.Dataset.from_parquet(config.dataset_path)
    dataset = dataset.filter(lambda r: len(r["logprobs"]) > 0)

    def collate_fn(batch):
        bs = len(batch)
        seq_len = max([len(r["token_ids"]) for r in batch])
        topk = len(batch[0]["logprobs"][0])

        input_ids = torch.full((bs, seq_len), 0, dtype=torch.long)
        attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
        topk_logprobs = torch.zeros((bs, seq_len - 1, topk), dtype=torch.float32)
        topk_logprobs_indices = torch.zeros((bs, seq_len - 1, topk), dtype=torch.long)
        label_mask = torch.zeros((bs, seq_len - 1), dtype=torch.float32)

        for idx in range(bs):
            item_seq_len = len(batch[idx]["token_ids"])
            item_prompt_len = batch[idx]["prompt_length"]
            input_ids[idx][:item_seq_len] = torch.as_tensor(batch[idx]["token_ids"])
            attention_mask[idx][:item_seq_len] = 1
            topk_logprobs[idx][item_prompt_len - 1 : item_seq_len - 1] = torch.as_tensor(batch[idx]["logprobs"])
            topk_logprobs_indices[idx][item_prompt_len - 1 : item_seq_len - 1] = torch.as_tensor(batch[idx]["logprobs_vocab_idx"])
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
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    eagle_module.train()

    for epoch in range(config.epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = compute_eagle_loss(eagle_module, base_model, batch, device=device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % config.log_every == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    save_eagle_checkpoint(eagle_module, config.output_dir, label="final")
