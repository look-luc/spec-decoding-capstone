import torch
import torch.nn as nn


class medusa_heads(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Initialize directly on target device & dtype to avoid FP32 memory spike
        self.linear = nn.Linear(
            in_features, out_features, bias=False, device="cpu", dtype=dtype
        )
        if device != "cpu" and device is not None:
            self.linear = self.linear.to(device)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_state)


class madusa(nn.Module):
    def __init__(self, base_model: nn.Module, num_heads: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.vocab_size = base_model.config.vocab_size
        hidden_size = base_model.config.hidden_size

        for param in self.base_model.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, self.vocab_size, bias=False)
            for _ in range(num_heads)
        ])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
                raise ValueError("Must provide either hidden_states or input_ids to Medusa module.")
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    **kwargs
                )
            hidden_states = outputs.hidden_states[-1]

        logits = torch.stack([head(hidden_states) for head in self.heads], dim=0)
        return logits
