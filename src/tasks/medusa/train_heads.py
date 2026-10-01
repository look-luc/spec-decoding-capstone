from typing import Any

from src.config.config import MedusaConfig


def _resolve_medusa_config(config: Any) -> MedusaConfig:
    """Extract or construct a valid MedusaConfig from an ExperimentConfig or MedusaConfig."""
    if isinstance(config, MedusaConfig):
        return config

    medusa_subcfg = getattr(config, "medusa", None)
    if isinstance(medusa_subcfg, MedusaConfig):
        return medusa_subcfg

    # Extract field fallbacks from top-level ExperimentConfig
    dataset_path = getattr(
        config, "dataset_path", getattr(config, "data_path", getattr(config, "dataset", None))
    )
    target_model = getattr(
        config, "target_model", getattr(config, "model", getattr(config, "base_model", None))
    )

    if not dataset_path:
        raise ValueError(
            "Configuration missing dataset path. Ensure 'dataset_path' or 'data_path' is set in ExperimentConfig."
        )

    return MedusaConfig(
        draft_model=getattr(config, "draft_model"),
        target_model=getattr(config, "target_model"),
        dataset_path=dataset_path,
        output_dir=getattr(config, "output_dir", "./output"),
        max_steps=getattr(config, "max_steps", getattr(config, "num_steps", 1000)),
        batch_size=getattr(config, "batch_size", 4),
        learning_rate=getattr(config, "learning_rate", 5e-5),
        grad_accum_steps=getattr(config, "grad_accum_steps", 8),
        num_heads=getattr(config, "num_heads", 4),
        eval_split_ratio=getattr(config, "eval_split_ratio", 0.05),
        warmup_ratio=getattr(config, "warmup_ratio", 0.1),
        lr_scheduler=getattr(config, "lr_scheduler", "cosine"),
        weight_decay=getattr(config, "weight_decay", 0.01),
        log_every=getattr(config, "log_every", 10),
        eval_every=getattr(config, "eval_every", 100),
        hf_repo_id=getattr(config, "hf_repo_id", None),
        wandb_project=getattr(config, "wandb_project", "spec-decoding"),
        language_code=getattr(config, "language_code", "en"),
        task=getattr(config, "task", "medusa"),
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
        entity=WANDB_ENTITY,
        config=asdict(cfg),
        group=group,
        job_type=f"distill-{cfg.task}",
        tags=tags,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    return run


def run_medusa_training(config: Any):
    cfg = _resolve_medusa_config(config)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info(f"Loading model: {cfg.target_model}")

    model, tokenizer = load_model(cfg.target_model, device=cfg.device)
    medusa_model = madusa(model, num_heads=cfg.num_heads)
    device = next(medusa_model.parameters()).device

    dataset = datasets.Dataset.from_parquet(cfg.dataset_path)
    dataset.set_format(type="torch", columns=["token_ids", "logprobs", "logprobs_vocab_idx"])
    dataset = dataset.filter(lambda r: len(r['logprobs']) > 0)
    repo_name = build_repo_name(cfg)
    logger.info(f"HF repo: {repo_name}")
    ...
