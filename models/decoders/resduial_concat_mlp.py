import torch
from torch import nn
from torch.nn import functional as F

from layer.base import MLP
from train_eval.utils import load_anchors


class DecoderResCat(nn.Module):
    """MLP & residual concat"""
    def __init__(self, args: dict):
        super(DecoderResCat, self).__init__()
        hidden_size = args['hidden_size']
        in_features = eval(args['in_features'])
        out_features = args['pred_length'] * args['pred_num'] * args[
            'pred_feature']
        self.pred_length = args['pred_length']
        self.pred_num = args['pred_num']
        self.pred_feature = args['pred_feature']
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)
        self.prob_fc = nn.Linear(hidden_size + in_features, self.pred_num)

        # anchors = load_anchors()[0]
        # self.anchors_tensor = nn.Parameter(anchors)

    def forward(self, hidden_states):
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states[:, 0, :]
        hidden_states = torch.cat(
            [hidden_states, self.mlp(hidden_states)], dim=-1)
        pred_probs = F.log_softmax(self.prob_fc(hidden_states), dim=-1)
        hidden_states = self.fc(hidden_states)
        outputs = hidden_states.view(hidden_states.shape[0], self.pred_num,
                                     self.pred_length, self.pred_feature)
        # outputs += self.anchors_tensor[:, :6, :].unsqueeze(0)
        return outputs, pred_probs
