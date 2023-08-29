import torch
import torch.nn as nn
from models.decoders.decoder import PredictionDecoder
from layer.base import MLP

from typing import Dict


class MLPdecx(PredictionDecoder):
    def __init__(self, args: Dict):
        input_size, hidden_size = \
            eval(args['in_features']), args['hidden_size']
        super(MLPdecx, self).__init__()
        self.pred_length = args['pred_length']
        self.pred_num = args['pred_num']
        self.pred_feature = args['pred_feature']
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            hidden_size, self.pred_length * self.pred_num * self.pred_feature)

    def forward(self, agg_encodings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP
        :param agg_encodings: aggregated context encodings
        :return: prediction result
        """
        x = self.linear1(agg_encodings)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(x.shape[0], self.pred_num, self.pred_length,
                   self.pred_feature)
        return x, torch.ones((x.shape[0], self.pred_num))


class MLPDecoder(PredictionDecoder):
    def __init__(self, args):
        super(MLPDecoder, self).__init__()
        self.hidden_size = args['hidden_size']
        in_features = eval(args['in_features'])
        self.layers = nn.ModuleList([
            MLP(in_features, self.hidden_size),
            MLP(self.hidden_size, self.hidden_size),
            MLP(self.hidden_size, self.hidden_size)
        ])
        self.pred_length = args['pred_length']
        self.pred_num = args['pred_num']
        self.pred_feature = args['pred_feature']
        self.fc = nn.Linear(
            self.hidden_size,
            self.pred_length * self.pred_num * self.pred_feature)

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        x = self.fc(x)
        # if self.pred_num == 1:
        #     x = x.view(x.shape[0], self.pred_length, self.pred_feature)
        # else:
        x = x.view(x.shape[0], self.pred_num, self.pred_length,
                   self.pred_feature)
        return x, torch.ones((x.shape[0], self.pred_num))
