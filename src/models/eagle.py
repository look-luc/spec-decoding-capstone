import torch
import torch.nn as nn


class EagleModule(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc_fusion = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_embedding(self, token_id: torch.Tensor) -> torch.Tensor:
        return self.embeddings(token_id)

    def forward(self, token_id: torch.Tensor, hidden_state: torch.Tensor):
        token_emb = self.get_embedding(token_id)
        fused = torch.cat([token_emb, hidden_state], dim=-1)
        projected = self.fc_fusion(fused)
        next_hidden = self.decoder_layer(projected)
        logits = self.lm_head(next_hidden)
        return next_hidden, logits
