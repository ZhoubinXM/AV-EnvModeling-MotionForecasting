from models.encoders.encoder import PredictionEncoder
from layer.deepfm import DeepFM
from layer.sub_graph import SubGraph
import torch
import torch.nn as nn
import math
from typing import Dict, Tuple


class DeppFMSubGraph(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        DeepFM for encoding ...
        SubGraph from VectorNet for encoding ...

        args to include:
            'veh_cate_fea_nuniqs': int Number of vehicle category
            'veh_nume_fea_size': int Length of ...
            'driver_cate_fea_nuniqs': int Number of ...
            'driver_nume_fea_size': int Length of ...
            'hidden_size': int Width of DeepFM output size and SubGraph input size
        """
        super().__init__()
        veh_cate_fea_nuniqs = args['veh_cate_fea_nuniqs']
        veh_nume_fea_size = args['veh_nume_fea_size']
        feature_size = args['feature_size']
        driver_cate_fea_nuniqs = args['driver_cate_fea_nuniqs']
        driver_nume_fea_size = args['driver_nume_fea_size']

        self.veh = DeepFM(veh_cate_fea_nuniqs, nume_fea_size=veh_nume_fea_size, emb_size=feature_size)
        self.driver = DeepFM(driver_cate_fea_nuniqs, nume_fea_size=driver_nume_fea_size, emb_size=feature_size)
        self.sub_graph = SubGraph(feature_size)

    def forward(self, veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
                attention_mask) -> Dict:
        # veh_cate_features = inputs[0]
        # veh_dense_features = inputs[1]
        # driver_cate_features = inputs[2]
        # driver_dense_features = inputs[3]
        # polylines = inputs[4]
        # polynum = inputs[5]
        # attention_mask = inputs[6]
        veh_vector = self.veh(veh_cate_features, veh_dense_features)  # (batch, feature_size)
        driver_vector = self.driver(driver_cate_features, driver_dense_features)  # (batch, feature_size)
        sub_vector = self.sub_graph(polylines, attention_mask, polynum)  # (all_poly_num, feature_size)

        # Return encoding
        encodings = {
            'vehicle_vector': veh_vector,
            'driver_vector': driver_vector,
            'subgraph_vector': sub_vector,
            'polynum': polynum
        }

        return encodings
