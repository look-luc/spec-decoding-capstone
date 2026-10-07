import logging
import math
import os
import time
from dataclasses import asdict
from typing import Any, Literal, cast

import datasets
import torch
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config.medusa_config import MedusaConfig
from src.data.dataset import assemble_dataset
from src.models.madusa import madusa
from src.utils import load_model

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TaskType = Literal["translation", "story_gen"]


def _model_short_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown_model"
    return model_name.strip("/").split("/")[-1]


def build_repo_name(cfg: Any) -> str:
    hf_repo_id = getattr(cfg, "hf_repo_id", None)
    if hf_repo_id:
        return hf_repo_id
    target_model = getattr(cfg, "target_model", None) or getattr(cfg, "draft_model", None)
    model_short = _model_short_name(target_model)
    lang = getattr(cfg, "language_code", "en")
    return f"{model_short}-medusa-{lang}"


def _resolve_medusa_config(config: Any) -> MedusaConfig:
    if isinstance(config, MedusaConfig):
        return config

    medusa_subcfg = getattr(config, "medusa", None)
    if isinstance(medusa_subcfg, MedusaConfig):
        return medusa_subcfg

    target_model = getattr(
        config, "target_model", getattr(config, "model", getattr(config, "base_model", None))
    )
    draft_model = getattr(config, "draft_model", None)

    raw_task = getattr(config, "task", "translation")
    task_val: TaskType = cast(
        TaskType,
        raw_task if raw_task in ("translation", "story_gen") else "translation",
    )

    return MedusaConfig(
        draft_model=draft_model,
        target_model=target_model,
        draft_model_type=getattr(config, "draft_model_type", "medusa"),
        decoding_mode=getattr(config, "decoding_mode", "greedy"),
        num_heads=getattr(config, "num_heads", 4),
        wandb_project=getattr(config, "wandb_project", "speculative decoding v2"),
        language_code=getattr(config, "language_code", "en"),
        task=task_val,
        learning_rate=getattr(config, "learning_rate", 2e-5),
        max_steps=getattr(config, "max_steps", 3000),
        grad_accum_steps=getattr(config, "grad_accum_steps", 100),
        device=getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"),
    )


def setup_wandb(config: Any):
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


def _build_scheduler(optimizer: optim.Optimizer, config: Any) -> LambdaLR:
    max_steps = getattr(config, "max_steps", 3000)
    warmup_ratio = getattr(config, "warmup_ratio", 0.1)
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    hf_schedule = getattr(config, "hf_schedule", "constant")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if hf_schedule == "heuristic":
            return 1.0
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        if hf_schedule == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(madusa_model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        # Freeze backbone forward pass to prevent tracking autograd activation graph
        with torch.no_grad():
            if hasattr(madusa_model, "get_hidden_states"):
                hidden_states = madusa_model.get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
            elif hasattr(madusa_model, "base_model"):
                outputs = madusa_model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[-1]
                del outputs  # Free all intermediate layer hidden state tensors
            else:
                outputs = madusa_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                del outputs

        hidden_states = hidden_states.detach()

        heads = getattr(madusa_model, "heads", None)
        num_heads = len(heads) if heads is not None else 4
        total_loss = 0.0

        for k in range(num_heads):
            head_logits = heads[k](hidden_states) if heads is not None else madusa_model.compute_head(hidden_states, k)

            preds = head_logits[:, :-(k + 1), :].contiguous().view(-1, head_logits.size(-1))
            targets = input_ids[:, (k + 1):].contiguous().view(-1)
            mask = label_mask[:, (k + 1):].contiguous().view(-1)

            loss_raw = torch.nn.functional.cross_entropy(preds, targets, reduction="none")
            weighted_loss = (loss_raw * mask).sum() / max(mask.sum(), 1)
            total_loss += weighted_loss

            del head_logits

    return total_loss / num_heads


@torch.no_grad()
def _compute_eval_loss(student: torch.nn.Module, eval_dataloader: DataLoader, device: torch.device) -> float:
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
    cfg = _resolve_medusa_config(config)

    output_dir = getattr(config, "output_dir", getattr(cfg, "output_dir", "./output"))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading model: {cfg.target_model}")

    model, tokenizer = load_model(cfg.target_model, device=cfg.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    num_embeddings = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    medusa_model = madusa(model, num_heads=cfg.num_heads)
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    medusa_model.to(device)

    # Directly assemble dataset in memory without saving to disk
    lang_code = getattr(config, "language_code", cfg.language_code)
    dataset_type = getattr(config, "dataset_type", "bi")
    max_samples = max(getattr(config, "max_samples"),6000)

    logger.info(f"Assembling dataset in memory for '{lang_code}' (type={dataset_type}, max_samples={max_samples})...")
    splits = assemble_dataset(
        lang_code=lang_code,
        type=dataset_type,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )

    train_dataset = splits["train"]
    eval_dataset = splits["test"] if "test" in splits else splits["train"].select([])

    if len(train_dataset) == 0:
        logger.error("Train dataset is empty.")
        return

    repo_name = build_repo_name(cfg)
    logger.info(f"HF repo: {repo_name}")
    logger.info(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval examples")

    def collate_fn(batch):
        bs = len(batch)
        max_seq_len_cap = getattr(config, "max_seq_len", 512)

        token_lists = []
        for r in batch:
            if "token_ids" in r and r["token_ids"] is not None:
                ids = r["token_ids"]
            elif "input_ids" in r and r["input_ids"] is not None:
                ids = r["input_ids"]
            elif "tokens" in r and r["tokens"] is not None:
                ids = r["tokens"]
            elif "text" in r and r["text"] is not None:
                ids = tokenizer.encode(r["text"], add_special_tokens=True)
            elif config.language_code in r and r[config.language_code] is not None:
                ids = tokenizer.encode(r[config.language_code], add_special_tokens=True)
            elif "English" in r and r["English"] is not None:
                ids = tokenizer.encode(r["English"], add_special_tokens=True)
            else:
                # General fallback: check any non-origin string column
                string_val = next((v for k, v in r.items() if k != "origin" and isinstance(v, str)), None)
                if string_val:
                    ids = tokenizer.encode(string_val, add_special_tokens=True)
                else:
                    raise KeyError(f"Batch item missing token IDs or text. Available keys: {list(r.keys())}")

            # Truncate to maximum length to prevent CUDA OOM
            token_lists.append(ids[:max_seq_len_cap])

        seq_len = max(len(ids) for ids in token_lists)
        input_ids = torch.full((bs, seq_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, seq_len), dtype=torch.long)
        label_mask = torch.zeros((bs, seq_len), dtype=torch.float32)

        for idx, ids in enumerate(token_lists):
            item_seq_len = len(ids)
            item_prompt_len = batch[idx].get("prompt_length", 1)

            input_ids[idx][:item_seq_len] = torch.as_tensor(ids)
            attention_mask[idx][:item_seq_len] = 1
            label_mask[idx][item_prompt_len:item_seq_len] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
        }

    # Reduced default batch size to 1 to prevent CUDA memory allocation issues
    batch_size = getattr(config, "batch_size", getattr(cfg, "batch_size", 1))
    dataloader = DataLoader(
        train_dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        num_workers=2,
    )
    eval_dataloader = DataLoader(
        eval_dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        num_workers=2,
    )

    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    weight_decay = getattr(config, "weight_decay", 0.01)
    param_groups = [
        {
            "params": [p for n, p in medusa_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in medusa_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=cfg.learning_rate)
        logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
    except ImportError as e:
        logger.error("bitsandbytes is not installed. 32-bit AdamW will exceed VRAM limit.")
        raise ImportError(
            "bitsandbytes is required to enforce 8-bit AdamW optimizer. "
            "Install it via `pip install bitsandbytes`."
        ) from e
    scheduler = _build_scheduler(optimizer, cfg)
    start_step = _restore_training_state(cfg, optimizer, scheduler, device)

    model_dtype = next(medusa_model.parameters()).dtype
    use_scaler = device.type == "cuda" and model_dtype == torch.float16
    scaler = GradScaler(device.type, enabled=use_scaler)

    step = start_step
    target_step = start_step + cfg.max_steps
    accum_count = 0
    log_accum_loss = 0.0
    log_micro_count = 0
    best_eval_loss = float("inf")
    start_time = time.time()
    epoch = 0

    log_every = getattr(config, "log_every", 10)
    eval_every = getattr(config, "eval_every", 100)

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

            if step % log_every == 0 and log_micro_count > 0:
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

            if step % eval_every == 0 and len(eval_dataset) > 0:
                eval_loss = _compute_eval_loss(medusa_model, eval_dataloader, device)
                logger.info(f"Step {step} | Eval loss: {eval_loss:.4f}")
                wandb.log({"eval/loss": eval_loss, "step": step})
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_checkpoint(
                        medusa_model, tokenizer, optimizer, output_dir, "best", repo_name,
                        push_to_hub=False, scheduler=scheduler,
                    )

            if step >= target_step:
                break

        epoch += 1
        if step < target_step:
            logger.info(f"Completed epoch {epoch}. Continuing to step {target_step}...")

    wandb.log({"eval/best_loss": best_eval_loss})

    hf_repo_id = getattr(config, "hf_repo_id", None)
    if hf_repo_id:
        logger.info(f"Training complete! Pushing final model to HF Hub: {repo_name}")
    else:
        logger.info("Training complete! Saving final checkpoint locally (HF Hub push disabled).")
    _save_checkpoint(
        medusa_model, tokenizer, optimizer, output_dir, "final", repo_name,
        push_to_hub=bool(hf_repo_id), scheduler=scheduler,
    )

    save_medusa_weights(medusa_model, output_dir, "medusa_heads.pt")


def save_medusa_weights(medusa_model: torch.nn.Module, output_dir: str, filename: str = "medusa_heads.pt"):
    """Saves only the trainable Medusa projection heads."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    if hasattr(medusa_model, "heads"):
        state_dict = medusa_model.heads.state_dict()
    else:
        state_dict = medusa_model.state_dict()

    torch.save(state_dict, save_path)
    logger.info(f"Saved Medusa head weights to: {save_path}")


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


def _save_checkpoint(
    student,
    tokenizer,
    optimizer,
    output_dir,
    label,
    repo_name=None,
    push_to_hub=False,
    scheduler=None,
):
    run_name = wandb.run.name if wandb.run else "medusa-run"
    path = os.path.join(output_dir, f"{run_name}-{label}.ckpt")
    os.makedirs(path, exist_ok=True)

    if hasattr(student, "save_pretrained"):
        student.save_pretrained(path)
    else:
        torch.save(student.state_dict(), os.path.join(path, "pytorch_model.bin"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)

    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
    logger.info(f"Saved checkpoint: {path}")

    if push_to_hub and repo_name:
        hub_repo = f"{repo_name}-{label}" if isinstance(label, int) else repo_name
        logger.info(f"Pushing to HF Hub: {hub_repo}")
        if hasattr(student, "push_to_hub"):
            student.push_to_hub(hub_repo, commit_message=f"Distilled model (step {label})")
        if tokenizer is not None and hasattr(tokenizer, "push_to_hub"):
            tokenizer.push_to_hub(hub_repo, commit_message=f"Tokenizer (step {label})")
        logger.info(f"Pushed: https://huggingface.co/{hub_repo}")


def grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
