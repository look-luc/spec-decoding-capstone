from dataclasses import dataclass
from typing import Literal

WANDB_PROJECT = "speculative decoding capstone"
WANDB_ENTITY = "lecs-general"

DEFAULT_MAX_BI = 7000
DEFAULT_MAX_MONO = 20500

@dataclass
class MedusaConfig:
    task: Literal['translation', 'story_gen']
    language_code: str

    target_model: str|None
    draft_model: str | None
    draft_model_type: Literal["medusa", "madusa"]
    decoding_mode: Literal["greedy", "sample"]
    num_heads:int = 4
    top_k: int = 0
    top_p: float = 0.0

    data_source: str = "tatoeba"
    dataset_path: str | None = None
    output_dir: str = "checkpoints"

    repetition_penalty: float = 1.1
    repetition_penalty_window: int = 16

    gamma: int = 5
    track_iterations: bool = False # If true, will log per-iteration of SD

    learning_rate:float=2e-5
    max_steps:int = 3000
    eval_every:int = 75
    log_every:int = 10
    grad_accum_steps:int = 100

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = "constant"

    max_samples: int = DEFAULT_MAX_BI
    max_samples_mono: int = DEFAULT_MAX_MONO
    max_new_tokens: int = 128
    story_seed: int | None = 0
    device: str = "auto"

    wandb_tag: str | None = None
    wandb_project: str = WANDB_PROJECT

    def __post_init__(self):
        if self.draft_model == "None":
            self.draft_model = None

        if isinstance(self.story_seed, str):
            self.story_seed = None if self.story_seed == "None" else int(self.story_seed)
