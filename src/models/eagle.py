import re

import torch
import torch.nn as nn


class EagleModule(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.hidden_norm = nn.RMSNorm(hidden_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(hidden_dim)
        self.fc_fusion = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(hidden_dim)

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    @staticmethod
    def infer_num_layers(state_dict: dict) -> int:
        layer_indices = []
        for key in state_dict.keys():
            match = re.match(r"^(?:module\.)?decoder_layers\.(\d+)\.", key)
            if match:
                layer_indices.append(int(match.group(1)))
        if not layer_indices:
            raise ValueError("Could not infer num_layers: no 'decoder_layers.<idx>.' keys found in state_dict.")
        return max(layer_indices) + 1

    def load_checkpoint_state_dict(self, state_dict: dict, strict: bool = True):
        cleaned = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        ckpt_layers = self.infer_num_layers(cleaned)
        if ckpt_layers != self.num_layers:
            raise ValueError(
                f"EagleModule layer mismatch: initialized with num_layers={self.num_layers}, "
                f"but checkpoint has num_layers={ckpt_layers}. Set num_layers={ckpt_layers} or use a matching checkpoint."
            )
        return self.load_state_dict(cleaned, strict=strict)

    def get_embedding(self, token_id: torch.Tensor) -> torch.Tensor:
        return self.embeddings(token_id)

    def forward(
        self,
        token_id: torch.Tensor,
        hidden_state: torch.Tensor,
        **kwargs
    ):
        token_emb = self.get_embedding(token_id)
        normed_hidden = self.hidden_norm(hidden_state)
        fused = torch.cat([token_emb, normed_hidden], dim=-1)
        projected = self.norm(self.fc_fusion(fused) + normed_hidden)

        seq_len = projected.size(1)
        tgt_mask = None
        if seq_len > 1:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=projected.device)

        hidden = projected
        for layer in self.decoder_layers:
            hidden = layer(
                hidden,
                hidden,
                tgt_mask=tgt_mask,
                tgt_is_causal=(seq_len > 1)
            )

        next_hidden = hidden
        logits = self.lm_head(next_hidden)
        if kwargs.get("use_cache", False) or "past_key_values" in kwargs:
            return logits, next_hidden, kwargs.get("past_key_values", None)

        return next_hidden, logits
