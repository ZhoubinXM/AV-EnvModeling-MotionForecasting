from torch import nn


class MLP(nn.Module):

    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        return hidden_states
