from dataclasses import dataclass
from typing import Literal

WANDB_PROJECT = "speculative decoding capstone"
WANDB_ENTITY = "lecs-general"

DEFAULT_MAX_BI = 7000
DEFAULT_MAX_MONO = 20500

@dataclass
class EagleConfig:
    task: Literal['translation', 'story_gen']
    language_code: str

    target_model: str = "Qwen/Qwen3.5-9B"
    draft_model: str | None = "Qwen/Qwen3.5-0.8B"
    draft_model_type: Literal["eagle", "Eagle"] = "eagle"
    decoding_mode: Literal["greedy", "sample"] = "greedy"
    tree_choices: str|list[int] = "custom"

    score_threshold: float = 0.02
    distillation_beta: float = 0.3
    top_k: int = 0
    top_p: float = 0.0

    repetition_penalty: float = 1.1
    repetition_penalty_window: int = 16

    gamma: int = 5
    num_heads: int = 8
    num_layers: int = 2
    track_iterations: bool = True

    ngram_n: int = 2

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = None

    max_samples: int = DEFAULT_MAX_BI
    max_samples_mono: int = DEFAULT_MAX_MONO
    max_new_tokens: int = 128
    max_seq_len:int = 512
    story_seed: int | None = 0
    device: str = "cuda"
    dataset_path: str | None = None

    learning_rate: float = 1e-4
    batch_size: int = 1
    grad_accum_steps: int = 4
    eval_every: int = 500
    max_steps: int = 3000
    warmup_steps: int = 200
    log_every: int = 10
    wandb_tag: str | None = "Eagle 7B Target"
    output_dir: str = "checkpoints/eagle_7b"
    wandb_project: str = WANDB_PROJECT

    def __post_init__(self):
        if self.draft_model == "None":
            self.draft_model = None

        if isinstance(self.story_seed, str):
            self.story_seed = None if self.story_seed == "None" else int(self.story_seed)
