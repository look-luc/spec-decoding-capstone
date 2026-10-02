from dataclasses import dataclass
from typing import Literal

WANDB_PROJECT = "speculative decoding v2"
WANDB_ENTITY = "lecs-general"

# Default dataset caps used in our experiments
DEFAULT_MAX_BI = 6000
DEFAULT_MAX_MONO = 20000

@dataclass
class EagleConfig:
    task: Literal['translation', 'story_gen']
    language_code: str

    target_model: str
    draft_model: str | None
    draft_model_type: Literal["eagle", "Eagle"]
    decoding_mode: Literal["greedy", "sample"]
    tree_choices: str|list[int] = "custom"
    num_heads: int = 4
    top_k: int = 0
    top_p: float = 0.0

    repetition_penalty: float = 1.1
    repetition_penalty_window: int = 16

    gamma: int = 5
    track_iterations: bool = False # If true, will log per-iteration of SD

    ngram_n: int = 2

    use_hf_assisted: bool = False
    hf_schedule: Literal["heuristic", "constant"] | None = None

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

        if self.draft_model_type == 'neural':
            assert self.gamma > 0
            assert self.draft_model is not None

        if isinstance(self.story_seed, str):
            self.story_seed = None if self.story_seed == "None" else int(self.story_seed)
