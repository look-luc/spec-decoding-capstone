import torch
import torch.nn as nn


class EagleModule(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_heads:int=8):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc_fusion = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_embedding(self, token_id: torch.Tensor) -> torch.Tensor:
        return self.embeddings(token_id)

    def forward(
        self,
        token_id: torch.Tensor,
        hidden_state: torch.Tensor,
        **kwargs
    ):
        token_emb = self.get_embedding(token_id)
        fused = torch.cat([token_emb, hidden_state], dim=-1)
        projected = self.fc_fusion(fused)

        seq_len = projected.size(1)
        tgt_mask = None
        if seq_len > 1:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=projected.device)

        next_hidden = self.decoder_layer(
            projected,
            projected,
            tgt_mask=tgt_mask,
            is_causal=(seq_len > 1)
        )
        logits = self.lm_head(next_hidden)
        if kwargs.get("use_cache", False) or "past_key_values" in kwargs:
            return logits, next_hidden, kwargs.get("past_key_values", None)

        return next_hidden, logits
