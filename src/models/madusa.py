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

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size

        base_param = next(self.base_model.parameters())
        target_dtype = base_param.dtype
        target_device = base_param.device

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.heads = nn.ModuleList(
            [
                medusa_heads(
                    self.hidden_size,
                    self.vocab_size,
                    device=target_device,
                    dtype=target_dtype,
                )
                for _ in range(int(num_heads))
            ]
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.heads.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.heads.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]
        medusa_logits = [head(last_hidden) for head in self.heads]

        return outputs.logits, medusa_logits, outputs.past_key_values
