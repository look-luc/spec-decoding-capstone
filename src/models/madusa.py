import torch
import torch.nn as nn


class medusa_heads(nn.module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.linear = nn.linear(in_features, out_features, bias=False)

    def forward(self, hidden_state):
        return self.linear(hidden_state)

class madusa(nn.module):
    def __init__(self, base_model_id, num_heads=4) -> None:
        super().__init__()
        self.base_model_id = base_model_id
