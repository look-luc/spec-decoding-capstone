import logging
import math
import os
from dataclasses import asdict
from typing import Any, Literal, cast

import datasets
import torch
import wandb

from src.config.medusa_config import MedusaConfig

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TaskType = Literal["translation", "story_gen"]

def _model_short_name(model_name: str | None) -> str:
    """Extract short model identifier from HuggingFace repo path or local path."""
    if not model_name:
        return "unknown_model"
    return model_name.strip("/").split("/")[-1]


def _resolve_medusa_config(config: Any) -> MedusaConfig:
    """Extract or construct a valid MedusaConfig from an ExperimentConfig or MedusaConfig."""
    if isinstance(config, MedusaConfig):
        return config

    medusa_subcfg = getattr(config, "medusa", None)
    if isinstance(medusa_subcfg, MedusaConfig):
        return medusa_subcfg

    # Extract field fallbacks safely from top-level ExperimentConfig
    dataset_path = getattr(
        config, "dataset_path", getattr(config, "data_path", getattr(config, "dataset", None))
    )
    target_model = getattr(
        config, "target_model", getattr(config, "model", getattr(config, "base_model", None))
    )
    draft_model = getattr(config, "draft_model", None)

    if not dataset_path:
        raise ValueError(
            "Configuration missing dataset path. Ensure 'dataset_path' or 'data_path' is set in ExperimentConfig."
        )

    raw_task = getattr(config, "task", "translation")
    task_val: TaskType = cast(
        TaskType,
        raw_task if raw_task in ("translation", "story_gen") else "translation",
    )

    return MedusaConfig(
        draft_model=draft_model,
        target_model=target_model,
        draft_model_type=getattr(config, "draft_model_type"),
        decoding_mode=getattr(config, "decoding_mode"),
        num_heads=getattr(config, "num_heads", 4),
        wandb_project=getattr(config, "wandb_project", "spec-decoding"),
        language_code=getattr(config, "language_code", "en"),
        task=task_val,
        learning_rate=getattr(config, "learning_rate",2e-5),
        max_steps=getattr(config, "max_steps"),
        grad_accum_steps=getattr(config, "grad_accum_steps"),
        device=getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"),
    )

def setup_wandb(config: Any):
    """Initialize wandb for distillation run tracking."""
    cfg = _resolve_medusa_config(config)

    model_short = _model_short_name(cfg.draft_model or cfg.target_model)
    group = f"distill_{model_short}__{cfg.language_code}"

    tags = [
        "distillation",
        cfg.language_code,
        cfg.task,
        model_short,
        f"lr={cfg.learning_rate}",
        f"steps={cfg.max_steps}",
        f"ga={cfg.grad_accum_steps}",
    ]

    run = wandb.init(
        project=cfg.wandb_project,
        entity=os.getenv("WANDB_ENTITY", None),
        config=asdict(cfg),
        group=group,
        job_type=f"distill-{cfg.task}",
        tags=tags,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    return run

def _build_scheduler(optimizer, config: MedusaConfig) -> LambdaLR:
    """Build LR scheduler with linear warmup then cosine/linear/constant decay."""
    warmup_steps = max(1, int(config.max_steps * config.warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if config.hf_schedule == "heuristic":
            return 1.0
        progress = (current_step - warmup_steps) / max(1, config.max_steps - warmup_steps)
        if config.hf_schedule == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # linear
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(madusa_model, batch, device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        base_logits, medusa_logits, _ = madusa_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        total_loss = 0.0
        for k in range(len(medusa_logits)):
            logprobs = torch.nn.functional.log_softmax(
                medusa_logits[:, :-(k+1), :].contiguous(),
                dim=-1
            )
            target_logprobs = batch["topk_logprobs"][:, (k+1):, :]
            target_indices = batch["topk_logprobs_indices"][:, (k + 1):, :]
            mask = batch["label_mask"][:, (k + 1):, :]

            model_logprobs = logprobs.gather(dim=-1, index=target_indices)

            head_loss = -(torch.exp(target_logprobs) * model_logprobs).sum(-1)
            weighted_loss = (head_loss * mask).sum() / max(mask.sum(), 1)

            total_loss += weighted_loss
    return total_loss / len(medusa_logits)


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

def run_medusa_training(config: Any):
    # Safely extract inner MedusaConfig if passed a top-level ExperimentConfig
    cfg = getattr(config, "medusa", config)

    output_dir = getattr(cfg, "output_dir", getattr(config, "output_dir", "./output"))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading model: {cfg.target_model}")

    model, tokenizer = load_model(cfg.target_model, device=cfg.device)
    medusa_model = madusa(model, num_heads=cfg.num_heads)
    device = next(medusa_model.parameters()).device

    assert cfg.dataset_path
    dataset = datasets.Dataset.from_parquet(cfg.dataset_path)
    dataset.set_format(type="torch", columns=["token_ids", "logprobs", "logprobs_vocab_idx"])
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.filter(lambda r: len(r['logprobs']) > 0)
    repo_name = build_repo_name(cfg)
    logger.info(f"HF repo: {repo_name}")

    if cfg.eval_split_ratio > 0 and len(dataset) > 1:
        split = dataset.train_test_split(
            test_size=cfg.eval_split_ratio, seed=42,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = dataset.select([])
    logger.info(
        f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval examples"
    )

    def collate_fn(batch):
        bs = len(batch)
        seq_len = max([len(r["token_ids"]) for r in batch])
        topk = len(batch[0]["logprobs"][0])

        input_ids = torch.full((bs, seq_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
        topk_logprobs = torch.zeros((bs, seq_len - 1, topk), dtype=medusa_model.dtype)
        topk_logprobs_indices = torch.zeros((bs, seq_len - 1, topk), dtype=torch.long)
        label_mask = torch.zeros((bs, seq_len - 1), dtype=medusa_model.dtype)

        for idx in range(bs):
            item_seq_len = len(batch[idx]["token_ids"])
            item_prompt_len = batch[idx]["prompt_length"]
            input_ids[idx][0:item_seq_len] = torch.as_tensor(batch[idx]["token_ids"])
            attention_mask[idx][0:item_seq_len] = 1
            topk_logprobs[idx][item_prompt_len-1:item_seq_len-1] = torch.as_tensor(batch[idx]["logprobs"])
            topk_logprobs_indices[idx][item_prompt_len-1:item_seq_len-1] = torch.as_tensor(batch[idx]["logprobs_vocab_idx"])
            label_mask[idx][item_prompt_len-1:item_seq_len-1] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
            "topk_logprobs": topk_logprobs,
            "topk_logprobs_indices": topk_logprobs_indices,
        }

    dataloader = DataLoader(
        train_dataset,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        num_workers=2
    )
    eval_dataloader = DataLoader(
        eval_dataset,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        num_workers=2
    )

    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    param_groups = [
        {
            "params": [p for n, p in medusa_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in medusa_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(param_groups, lr=cfg.learning_rate)
    scheduler = _build_scheduler(optimizer, cfg)
    start_step = _restore_training_state(cfg, optimizer, scheduler, device)

    use_scaler = device.type == "cuda" and medusa_model.dtype == torch.float16
    scaler = GradScaler(device.type, enabled=use_scaler)

    step = start_step
    target_step = start_step + cfg.max_steps
    accum_count = 0
    log_accum_loss = 0.0
    log_micro_count = 0
    best_eval_loss = float("inf")
    start_time = time.time()
    epoch = 0

    logger.info(f"Training from step {start_step} to {target_step} (optimizer steps)")

    while step < target_step:
        for batch in dataloader:
            if step >= target_step:
                break

            loss = compute_loss(medusa_model, batch, device)
            if torch.isnan(loss):
                logger.warning(f"Step {step}, micro-batch NaN — skipping accumulation window")
                optimizer.zero_grad()
                accum_count = 0
                continue

            scaler.scale(loss / cfg.grad_accum_steps).backward()
            accum_count += 1
            log_accum_loss += loss.item()
            log_micro_count += 1

            if accum_count < cfg.grad_accum_steps:
                continue

            scaler.unscale_(optimizer)
            unclipped_grad_norm = grad_norm(medusa_model)
            torch.nn.utils.clip_grad_norm_(medusa_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            accum_count = 0
            step += 1

            if step % cfg.log_every == 0 and log_micro_count > 0:
                avg_loss = log_accum_loss / log_micro_count
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step} | Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
                )
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                    "train/grad_norm": unclipped_grad_norm,
                    "step": step,
                })
                log_accum_loss = 0.0
                log_micro_count = 0
                start_time = time.time()

            if step % cfg.eval_every == 0 and len(eval_dataset) > 0:
                eval_loss = _compute_eval_loss(medusa_model, eval_dataloader, device)
                logger.info(f"Step {step} | Eval loss: {eval_loss:.4f}")
                wandb.log({"eval/loss": eval_loss, "step": step})
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_checkpoint(
                        medusa_model, tokenizer, optimizer, cfg.output_dir, "best", repo_name,
                        push_to_hub=False, scheduler=scheduler,
                    )

            if step >= target_step:
                break

        epoch += 1
        if step < target_step:
            logger.info(f"Completed epoch {epoch}. Continuing to step {target_step}...")

    wandb.log({"eval/best_loss": best_eval_loss})

    if cfg.hf_repo_id:
        logger.info(f"Training complete! Pushing final model to HF Hub: {repo_name}")
    else:
        logger.info("Training complete! Saving final checkpoint locally (HF Hub push disabled).")
    _save_checkpoint(
        medusa_model, tokenizer, optimizer, cfg.output_dir, "final", repo_name,
        push_to_hub=bool(cfg.hf_repo_id), scheduler=scheduler,
    )
    wandb.finish()
    save_medusa_weights(medusa_model, cfg.output_dir, "medusa_heads.pt")


def save_medusa_weights(medusa_model, output_dir: str, filename: str = "medusa_heads.pt"):
    """Saves only the trainable Medusa projection heads."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    if hasattr(medusa_model, "heads"):
        state_dict = medusa_model.heads.state_dict()
    else:
        state_dict = medusa_model.state_dict()

    torch.save(state_dict, save_path)
    logger.info(f"Saved Medusa head weights to: {save_path}")


def _restore_training_state(config: MedusaConfig, optimizer, scheduler, device) -> int:
    """Restore optimizer and scheduler state from checkpoint; return starting step."""
    start_step = 0
    if config.resume_from:
        checkpoint_name = os.path.basename(config.resume_from)
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            logger.info(f"Loading optimizer state from {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        else:
            logger.warning("No optimizer state found — learning rates will reset")

        sched_path = os.path.join(config.resume_from, "scheduler.pt")
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


def _save_checkpoint(student, tokenizer, optimizer, output_dir, label,
                     repo_name=None, push_to_hub=False, scheduler=None):
    """Save model, tokenizer, optimizer, and scheduler state; optionally push to HF Hub."""
    run_name = wandb.run.name if wandb.run else "medusa-run"
    path = os.path.join(output_dir, f"{run_name}-{label}.ckpt")
    os.makedirs(path, exist_ok=True)

    student.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
    logger.info(f"Saved checkpoint: {path}")

    if push_to_hub and repo_name:
        hub_repo = f"{repo_name}-{label}" if isinstance(label, int) else repo_name
        logger.info(f"Pushing to HF Hub: {hub_repo}")
        student.push_to_hub(hub_repo, commit_message=f"Distilled model (step {label})")
        tokenizer.push_to_hub(hub_repo, commit_message=f"Tokenizer (step {label})")
        logger.info(f"Pushed: https://huggingface.co/{hub_repo}")


def grad_norm(model):
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm**0.5
    return grad_norm
