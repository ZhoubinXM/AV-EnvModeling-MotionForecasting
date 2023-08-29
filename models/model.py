import torch
import torch.nn as nn
import models.encoders.encoder as enc
import models.aggregators.aggregator as agg
import models.decoders.decoder as dec
from typing import Dict, Tuple, Optional


class ADMS(nn.Module):
    """
    ADMS prediction model
    """
    def __init__(self, encoder: enc.PredictionEncoder,
                 aggregator: agg.PredictionAggregator,
                 decoder: dec.PredictionDecoder):
        """
        Initializes model for ADMS prediction task
        """
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    def forward(self, batch: Dict, device: Optional[int] = 0) -> torch.Tensor:
        """
        Forward pass for prediction model
        :param inputs: Dictionary with ...
            ...
        :return outputs: Prediction Result
        """
        encodings = self.encoder(batch, device)
        # encodings = self.aggregator(encodings, device)
        outputs = self.decoder(encodings)

        return outputs
