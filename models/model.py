import torch
import torch.nn as nn
import models.encoders.encoder as enc
import models.aggregators.aggregator as agg
import models.decoders.decoder as dec
from typing import Dict, Tuple


class ADMS(nn.Module):
    """
    ADMS prediction model
    """

    def __init__(self, encoder: enc.PredictionEncoder, aggregator: agg.PredictionAggregator, decoder: dec.PredictionDecoder):
        """
        Initializes model for ADMS prediction task
        """
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    def preprocess_inputs(self, inputs: Dict) -> Tuple:
        veh_cate_features = inputs['veh_cate_features']
        veh_dense_features = inputs['veh_dense_features']
        driver_cate_features = inputs['driver_cate_features']
        driver_dense_features = inputs['driver_dense_features']
        polylines = inputs['polylines']
        attention_mask = inputs['attention_mask']
        polynum = inputs['polynum']
        return (veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
                attention_mask)

    def forward(self, veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
                attention_mask) -> torch.Tensor:
        """
        Forward pass for prediction model
        :param inputs: Dictionary with ...
            ...
        :return outputs: Prediction Result
        """
        encodings = self.encoder(veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines,
                                 polynum, attention_mask)
        agg_encodings = self.aggregator(encodings)
        outputs = self.decoder(agg_encodings)

        return outputs
