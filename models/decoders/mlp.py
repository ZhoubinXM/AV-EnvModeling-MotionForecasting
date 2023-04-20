import torch
import torch.nn as nn
from models.decoders.decoder import PredictionDecoder

from typing import Dict


class MLP(PredictionDecoder):

    def __init__(self, args: Dict):
        input_size, hidden_size, output_size = \
            args['feature_size'], args['hidden_size'], args['output_size']
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

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
        return x
