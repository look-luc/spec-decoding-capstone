import torch
import torch.nn as nn


class MedusaHead(nn.Module):
    """
    A 2-layer MLP head with a residual connection and SiLU non-linearity
    for higher-capacity multi-token drafting.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # Intermediate projection layer keeping hidden_size dimension
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype),
            nn.SiLU(),
        )
        # Final projection to vocabulary size
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Residual connection over the intermediate hidden projection
        residual = hidden_states
        hidden_states = self.mlp(hidden_states) + residual
        return self.proj(hidden_states)


class madusa(nn.Module):
    def __init__(self, base_model: nn.Module, num_heads: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.vocab_size = base_model.config.vocab_size
        hidden_size = base_model.config.hidden_size

        for param in self.base_model.parameters():
            param.requires_grad = False

        base_dtype = getattr(base_model, "dtype", torch.float32)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Instantiate MLP-based Medusa heads
        self.heads = nn.ModuleList([
            MedusaHead(
                hidden_size=hidden_size,
                vocab_size=self.vocab_size,
                dtype=base_dtype,
            )
            for _ in range(num_heads)
        ])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Runs the base model forward pass to extract last hidden states."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    @property
    def dtype(self) -> torch.dtype:
        return next(self.heads.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.heads.parameters()).device

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        if hidden_states is None:
            if input_ids is None:
                raise ValueError(
                    "Must provide either hidden_states or input_ids to Medusa"
                    " module."
                )
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids, output_hidden_states=True, **kwargs
                )
            hidden_states = outputs.hidden_states[-1]

        logits = torch.stack(
            [head(hidden_states) for head in self.heads], dim=0
        )
        return logits
