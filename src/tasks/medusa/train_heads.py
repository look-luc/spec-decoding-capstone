import logging
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
