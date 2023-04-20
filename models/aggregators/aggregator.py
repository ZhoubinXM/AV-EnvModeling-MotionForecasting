import torch
import torch.nn as nn
import abc
from typing import Dict, Union


class PredictionAggregator(nn.Module):
    """
    Base class for context aggregators for ADMS prediction task.
    Aggregates a set of context (lane, surrounding agent, driver status) encodings and
    outputs either a single aggregated context vector
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, encodings: Dict) -> Union[Dict, torch.Tensor]:
        """
        Forward pass for prediction aggregator
        :param encodings: Dictionary with driver and context encodings
        :return agg_encoding: Aggregated context encoding
        """
        raise NotImplementedError()
