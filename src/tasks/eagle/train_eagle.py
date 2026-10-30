import logging
import math
import os
import time
from dataclasses import asdict
from typing import Any

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import assemble_dataset
from src.models.eagle import EagleModule
from src.spec_decode_eagle import spec_decode_eagle

logger = logging.getLogger(__name__)


def compute_eagle_loss(
    eagle_module: nn.Module,
    base_model: nn.Module,
    batch: dict,
    device: torch.device = None,
    top_k: int = 40,
    beta: float = 0.3,          # Lower beta to lean more into soft KL distillation
    temperature: float = 2.0    # Softens the target distribution
) -> torch.Tensor:
    if device is None:
        device = next(base_model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    label_mask = batch["label_mask"].to(device)

    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        with torch.no_grad():
            base_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            base_hidden_states = base_outputs.hidden_states[-1]
            base_logits = base_outputs.logits

            greedy_targets = torch.argmax(base_logits, dim=-1)

            # Apply temperature scaling to target logits
            scaled_base_logits = base_logits.float() / temperature
            target_log_softmax = torch.nn.functional.log_softmax(scaled_base_logits, dim=-1)
            target_logprobs, target_indices = torch.topk(target_log_softmax, k=top_k, dim=-1)

        pred_hidden, pred_logits = eagle_module(
            token_id=input_ids[:, :-1],
            hidden_state=base_hidden_states[:, :-1, :],
        )

        seq_len = pred_logits.size(1)
        shifted_target_logprobs = target_logprobs[:, 1 : 1 + seq_len, :]
        matched_target_indices = target_indices[:, 1 : 1 + seq_len, :]
        shifted_greedy_targets = greedy_targets[:, 1 : 1 + seq_len]
        step_mask = label_mask[:, :seq_len]

        # Apply temperature scaling to student logits
        scaled_pred_logits = pred_logits.float() / temperature
        target_probs = torch.softmax(shifted_target_logprobs, dim=-1)
        student_logprobs = torch.nn.functional.log_softmax(scaled_pred_logits, dim=-1)

        gathered_student_logprobs = student_logprobs.gather(dim=-1, index=matched_target_indices)
        # Scale KL loss by T^2 to maintain gradient magnitude
        kl_loss = -(target_probs * gathered_student_logprobs).sum(dim=-1) * (temperature ** 2)

        ce_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            shifted_greedy_targets.reshape(-1),
            reduction="none"
        ).reshape(shifted_greedy_targets.shape)

        combined_loss = ((1.0 - beta) * kl_loss) + (beta * ce_loss)
        mask_sum = step_mask.sum().clamp(min=1)
        loss = (combined_loss * step_mask).sum() / mask_sum

    return loss


@torch.no_grad()
def _compute_eval_loss(student: torch.nn.Module, base_model: torch.nn.Module, eval_dataloader: DataLoader, device: torch.device) -> float:
    student.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_loss = 0.0
    count = 0

    max_eval_batches = 50
    for i, batch in enumerate(eval_dataloader):
        if i >= max_eval_batches:
            break
        loss = compute_eagle_loss(student, base_model, batch, device)
        total_loss += loss.item()
        count += 1

    student.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / max(count, 1)


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


def log_speculative_metrics(metrics: dict, step: int = None, use_wandb: bool = True):
    """
    Logs all speculative decoding metrics, including speedup coefficient,
    acceptance rates, octile position performance, and execution variances.
    """
    speedup = metrics.get("speed_up_factor", 1.0)
    acceptance_rate = metrics.get("acceptance_rate", 0.0)
    cost_ratio = metrics.get("inference_cost_ratio", 0.0)
    toks_per_sec = metrics.get("toks_per_sec", 0.0)
    matched = metrics.get("matched_tokens", 0)
    drafted = metrics.get("draft_tokens", 0)
    total_tokens = metrics.get("generated_tokens", 0)
    termination = metrics.get("termination_reason", "unknown")

    log_msg = (
        f"\n{'='*20} Speculative Decoding Metrics {'='*20}\n"
        f"  • Speed-up Coefficient (Factor) : {speedup:.4f}x\n"
        f"  • Token Acceptance Rate         : {acceptance_rate * 100:.2f}% ({matched}/{drafted})\n"
        f"  • Generation Speed              : {toks_per_sec:.2f} tokens/sec\n"
        f"  • Inference Cost Ratio (c)      : {cost_ratio:.4f}\n"
        f"  • Total Generated Tokens        : {total_tokens}\n"
        f"  • Termination Reason            : {termination}\n"
    )

    if "average_draft_time" in metrics:
        log_msg += (
            f"  • Avg Draft Time                : {metrics['average_draft_time']*1000:.2f} ms "
            f"(var: {metrics.get('draft_time_variance', 0):.4f})\n"
            f"  • Avg Verifier Time             : {metrics['average_verifier_time']*1000:.2f} ms "
            f"(var: {metrics.get('verifier_time_variance', 0):.4f})\n"
        )

    octile_acc = metrics.get("octile_position_acceptance", [])
    if octile_acc:
        log_msg += "  • Octile Position Acceptances   : " + ", ".join([f"P{i+1}: {val*100:.1f}%" if val is not None else f"P{i+1}: N/A" for i, val in enumerate(octile_acc)]) + "\n"

    log_msg += "=" * 65
    logger.info(log_msg)

    if use_wandb and wandb.run is not None:
        wandb_payload = {
            "spec_dec/speed_up_factor": speedup,
            "spec_dec/acceptance_rate": acceptance_rate,
            "spec_dec/toks_per_sec": toks_per_sec,
            "spec_dec/cost_ratio": cost_ratio,
            "spec_dec/total_generated_tokens": total_tokens,
        }
        if "average_draft_time" in metrics:
            wandb_payload["spec_dec/avg_draft_time_ms"] = metrics["average_draft_time"] * 1000
            wandb_payload["spec_dec/avg_verifier_time_ms"] = metrics["average_verifier_time"] * 1000

        for i, val in enumerate(octile_acc):
            if val is not None:
                wandb_payload[f"spec_dec/octile_acceptance_pos_{i+1}"] = val

        if step is not None:
            wandb.log(wandb_payload, step=step)
        else:
            wandb.log(wandb_payload)


def run_eagle_training(config, train_dataset, eval_dataset=None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing EAGLE training on device: {device}")

    # 1. Load Target Model (Frozen in eval mode)
    logger.info(f"Loading target model: {config.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        config.target_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(config.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_hidden_dim = target_model.config.hidden_size
    vocab_size = target_model.config.vocab_size

    # 2. Initialize EagleModule draft network
    logger.info("Initializing EagleModule architecture...")
    eagle_module = EagleModule(
        vocab_size=vocab_size,
        embed_dim=getattr(config, "embed_dim", 896),
        hidden_dim=target_hidden_dim,
        num_heads=getattr(config, "num_heads", 8),
        num_layers=getattr(config, "num_layers", 2)
    ).to(device)

    eagle_module.train()

    # 3. Setup Optimizer, Scheduler, and Scaler
    optimizer = bnb.optim.AdamW8bit(
        [p for p in eagle_module.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01
    )

    scheduler = _build_scheduler(optimizer, config)
    start_step = _restore_training_state(config, optimizer, scheduler, device)

    scaler = GradScaler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=getattr(train_dataset, "collate_fn", None)
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=getattr(eval_dataset, "collate_fn", None)
        )

    os.makedirs(config.output_dir, exist_ok=True)
    global_step = start_step
    best_eval_loss = float('inf')

    logger.info(f"Starting training loop from step {global_step}...")
    for epoch in range(getattr(config, "num_epochs", 1)):
        logger.info(f"--- Epoch {epoch + 1} ---")
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(dtype=torch.float16):
                with torch.no_grad():
                    target_outputs = target_model(
                        input_ids,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    hidden_states = target_outputs.hidden_states[-1]

                shift_input_ids = input_ids[:, :-1]
                shift_hidden = hidden_states[:, :-1, :]
                shift_labels = labels[:, 1:]

                _, logits = eagle_module(shift_input_ids, shift_hidden)

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, vocab_size), shift_labels.reshape(-1))

                if config.grad_accum_steps > 1:
                    loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(eagle_module.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Step the learning rate scheduler
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every == 0:
                    current_loss = loss.item() * config.grad_accum_steps
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"Global Step {global_step} | Loss: {current_loss:.4f} | LR: {current_lr:.6f}")
                    if wandb.run is not None:
                        wandb.log({"train/loss": current_loss, "train/lr": current_lr}, step=global_step)

                if global_step >= config.max_steps:
                    break

        # 4. Evaluate and save ONLY the best checkpoint using _save_checkpoint
        if eval_loader is not None:
            eval_loss = _compute_eval_loss(eagle_module, target_model, eval_loader, device)
            logger.info(f"Epoch {epoch + 1} Evaluation Loss: {eval_loss:.4f}")
            if wandb.run is not None:
                wandb.log({"eval/loss": eval_loss}, step=global_step)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                logger.info(f"⭐ New best model found! (Eval Loss: {best_eval_loss:.4f}). Saving checkpoint...")
                _save_checkpoint(
                    student=eagle_module,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    output_dir=config.output_dir,
                    label="best",
                    repo_name=getattr(config, "hf_repo_id", None),
                    push_to_hub=getattr(config, "push_to_hub", False),
                    scheduler=scheduler
                )
                # Also save a copy as eagle_module_best.pt for generation.py compatibility
                torch.save(eagle_module.state_dict(), os.path.join(config.output_dir, "eagle_module_best.pt"))
        else:
            # Fallback if no eval loader
            _save_checkpoint(
                student=eagle_module,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=config.output_dir,
                label="best",
                repo_name=getattr(config, "hf_repo_id", None),
                push_to_hub=getattr(config, "push_to_hub", False),
                scheduler=scheduler
            )
            torch.save(eagle_module.state_dict(), os.path.join(config.output_dir, "eagle_module_best.pt"))

        if global_step >= config.max_steps:
            break

    # 5. Post-Training Audit & Metrics Evaluation
    logger.info("Running post-training speculative correctness audit & speedup coefficient evaluation...")
    if eval_loader is not None:
        audit_speculative_correctness(target_model, eagle_module, tokenizer, eval_loader, device=device, max_samples=20)

        eagle_module.eval()
        with torch.no_grad():
            for batch in eval_loader:
                sample_prompt = batch["input_ids"][:1, :5].to(device)
                _, metrics = spec_decode_eagle(
                    target_model=target_model,
                    eagle_module=eagle_module,
                    tokenizer=tokenizer,
                    input_ids=sample_prompt,
                    mode=config.decoding_mode,
                    max_new_tokens=config.max_new_tokens,
                    tree_choices=getattr(config, "tree_choices", "custom"),
                    device=device
                )
                log_speculative_metrics(metrics, step=global_step, use_wandb=True)
                break

    logger.info("EAGLE training and evaluation completed successfully!")
    return eagle_module

def audit_speculative_correctness(target_model, eagle_module, tokenizer, eval_dataloader, device="cuda", max_samples=50):
    target_model.eval()
    eagle_module.eval()

    total_samples = 0
    exact_matches = 0

    print("Starting speculative decoding correctness audit...")
    with torch.no_grad():
        for batch in eval_dataloader:
            if batch is None:
                continue
            input_ids = batch["input_ids"][:, :5].to(device)  # Use prompt prefix

            for i in range(input_ids.size(0)):
                if total_samples >= max_samples:
                    break

                prompt = input_ids[i:i+1]

                # 1. Target-only greedy baseline generation
                target_out = target_model.generate(
                    prompt,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                # 2. Speculative decoding generation
                spec_out, _ = spec_decode_eagle(
                    target_model=target_model,
                    eagle_module=eagle_module,
                    tokenizer=tokenizer,
                    input_ids=prompt,
                    mode="greedy",
                    max_new_tokens=64,
                    tree_choices="balance_1",
                    device=device
                )

                min_len = min(target_out.size(1), spec_out.size(1))
                target_trimmed = target_out[:, :min_len]
                spec_trimmed = spec_out[:, :min_len]

                match = torch.equal(target_trimmed, spec_trimmed)
                if match:
                    exact_matches += 1
                else:
                    print(f"\n[Mismatch] Sample {total_samples}")
                    print(f"Target text: {tokenizer.decode(target_out[0], skip_special_tokens=True)}")
                    print(f"Spec text:   {tokenizer.decode(spec_out[0], skip_special_tokens=True)}")

                total_samples += 1
            if total_samples >= max_samples:
                break

    agreement_rate = exact_matches / total_samples if total_samples > 0 else 0.0
    print(f"\nCorrectness Audit Result: {exact_matches}/{total_samples} exact matches ({agreement_rate * 100:.1f}%)")
    if wandb.run is not None:
        wandb.log({"eval/correctness_agreement_rate": agreement_rate})
    return agreement_rate

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

def build_repo_name(cfg: Any) -> str:
    hf_repo_id = getattr(cfg, "hf_repo_id", None)
    if hf_repo_id:
        return hf_repo_id
    target_model = getattr(cfg, "target_model", None) or getattr(cfg, "draft_model", None)
    model_short = _model_short_name(target_model)
    lang = getattr(cfg, "language_code", "en")
    return f"{model_short}-medusa-{lang}"

def _model_short_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown_model"
    return model_name.strip("/").split("/")[-1]

def setup_wandb(config: Any):
    model_short = _model_short_name(config.draft_model or config.target_model)
    group = f"distill_{model_short}__{config.language_code}"

    tags = [
        "distillation",
        config.language_code,
        config.task,
        model_short,
        f"lr={config.learning_rate}",
        f"steps={config.max_steps}",
        f"ga={config.grad_accum_steps}",
    ]

    run = wandb.init(
        project=config.wandb_project,
        entity=os.getenv("WANDB_ENTITY", None),
        config=asdict(config),
        group=group,
        job_type=f"distill-{config.task}",
        tags=tags,
        name=f"{config.draft_model_type} {config.max_steps} {config.tree_choices} {config.decoding_mode}",
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    return run
