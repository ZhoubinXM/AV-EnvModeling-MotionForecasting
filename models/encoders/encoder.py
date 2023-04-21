import torch.nn as nn
import abc

from typing import Dict, Tuple, Optional, List


class PredictionEncoder(nn.Module):
    """
    Base class for encoders for ADMS prediction task
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, inputs: List[Dict], device: Optional[int] = 0) -> Dict:
        """
        Abstract method for forward pass. Returns dictionary of encodings.

        :param inputs: Dictionary with ...
        :return encodings: Dictionary with input encodings
        """
        raise NotImplementedError()
