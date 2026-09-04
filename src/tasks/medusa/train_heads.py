import logging
import math
import os
import time
from dataclasses import asdict

import datasets
import torch
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.config import WANDB_ENTITY, DistillConfig
from src.utils import load_model

logger = logging.getLogger(__name__)


def _model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def build_repo_name(config: DistillConfig) -> str:
    student = _model_short_name(config.student_model)
    teacher = _model_short_name(config.teacher_model)
    name = f"{config.language_code}-{config.task}-{teacher}-{student}"
    if config.hf_repo_id:
        return f"{config.hf_repo_id}/{name}"
    return name


def setup_wandb(config: DistillConfig):
    """Initialize wandb for distillation run tracking."""
    teacher_short = _model_short_name(config.teacher_model)
    student_short = _model_short_name(config.student_model)

    group = f"distill_{teacher_short}__{config.language_code}"

    tags = [
        "distillation",
        config.language_code,
        config.task,
        teacher_short,
        student_short,
        f"lr={config.learning_rate}",
        f"steps={config.max_steps}",
        f"ga={config.grad_accum_steps}",
    ]

    run = wandb.init(
        project=config.wandb_project,
        entity=WANDB_ENTITY,
        config=asdict(config),
        group=group,
        job_type=f"distill-{config.task}",
        tags=tags,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    return run


def _build_scheduler(optimizer, config: DistillConfig) -> LambdaLR:
    """Build LR scheduler with linear warmup then cosine/linear/constant decay."""
    warmup_steps = max(1, int(config.max_steps * config.warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if config.lr_scheduler == "constant":
            return 1.0
        progress = (current_step - warmup_steps) / max(1, config.max_steps - warmup_steps)
        if config.lr_scheduler == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # linear
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(student, batch, device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    topk_logprobs = batch["topk_logprobs"].to(device, non_blocking=True)
    topk_logprobs_idx = batch["topk_logprobs_indices"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device, non_blocking=True)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :].contiguous(), dim=-1)
        student_logprobs = logprobs.gather(dim=-1, index=topk_logprobs_idx)
        loss = -(torch.exp(topk_logprobs) * student_logprobs).sum(-1)
        loss = (loss * label_mask).sum() / label_mask.sum().clamp(min=1)
    return loss

@torch.no_grad()
def _compute_eval_loss(student, eval_dataloader, device) -> float:
    """Run a forward pass over the eval split and return average loss."""
    student.eval()
    total_loss = 0.0
    count = 0
    for batch in eval_dataloader:
        loss = compute_loss(student, batch, device)
        total_loss += loss.item()
        count += 1
    student.train()
    return total_loss / max(count, 1)
