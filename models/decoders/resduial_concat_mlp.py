import torch
from torch import nn
from torch.nn import functional as F

from layer.base import MLP


class DecoderResCat(nn.Module):
    """MLP & residual concat"""
    def __init__(self, args: dict):
        super(DecoderResCat, self).__init__()
        hidden_size = args['hidden_size']
        in_features = args['in_features']
        out_features = args['out_features']
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = torch.cat(
            [hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        pred_probs = F.log_softmax(hidden_states[:, -6:], dim=-1)
        outputs = outputs.view(hidden_states.shape[0], 6, 30, 2)
        return outputs, pred_probs
