import torch
import torch.nn as nn
import abc

from typing import Union, Dict


class PredictionDecoder(nn.Module):
    """
    Base class for decoders for ADMS prediction task.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, encoding: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Forward pass for ADMS decoder
        :param encoding: Context encoding
        :return outputs: Prediction Header Result
        """
        raise NotImplementedError()
