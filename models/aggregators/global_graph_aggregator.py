import torch
from models.aggregators.aggregator import PredictionAggregator
from layer.global_graph import GlobalGraph
from typing import Dict
import global_var


class GlobalGraphAggregator(PredictionAggregator):
    """
    Concatenates all context encodings.
    """

    def __init__(self, args: Dict):
        super().__init__()
        self.feature_size = args['feature_size']
        self.global_graph = GlobalGraph(self.feature_size)

    def forward(self, encodings: Dict) -> torch.Tensor:
        if self.training:
            return self._train(encodings)
        else:
            # train_res = self._train(encodings)
            # inference_res = self._inference(encodings)
            # if torch.sum(train_res - inference_res) != 0:
            #     print("error")
            # return self._train(encodings)
            return self._inference(encodings)

    def _train(self, encodings: Dict) -> torch.Tensor:
        """
        Forward pass for global graph aggregator
        """
        sub_vector = encodings['subgraph_vector']
        veh_vector = encodings['vehicle_vector']
        driver_vector = encodings['driver_vector']
        poly_num = encodings['polynum']
        poly_num_expand = poly_num + 0
        global_graph_batch = poly_num_expand.shape[0]
        max_poly_num = torch.max(poly_num_expand)
        device = sub_vector.device
        global_graph_input = torch.zeros([global_graph_batch, max_poly_num, self.feature_size], device=device)
        global_graph_attention_mask = torch.zeros([global_graph_batch, max_poly_num, max_poly_num], device=device)
        idx = 0
        for i, length in enumerate(poly_num_expand):
            # global_graph_input[i, 0, :] = veh_vector[i, :]
            # global_graph_input[i, 1, :] = driver_vector[i, :]
            global_graph_input[i, :length, :] = sub_vector[idx:idx + length - 0]
            global_graph_attention_mask[i, :length, :length] = torch.ones([length, length], device=device)
            idx += (length - 0)
        global_vector = self.global_graph(global_graph_input, global_graph_attention_mask)
        output = torch.cat([driver_vector, global_vector], dim=1)

        # global_var.set_value('veh_vector', veh_vector.detach().cpu().numpy())
        global_var.set_value('driver_vector', driver_vector.detach().cpu().numpy())
        global_var.set_value('global_vector', global_vector.detach().cpu().numpy())


        return output

    def _inference(self, encodings: Dict) -> torch.Tensor:
        # **shape** batch size = 1
        # polylines: [256, 19, self.hidden_size]
        # poly_num: [1,]
        # attention_mask: [256, 19, self.hidden_size // 2]
        # global_vector: [1, self.hidden_size]
        sub_vector = encodings['subgraph_vector']
        veh_vector = encodings['vehicle_vector']
        driver_vector = encodings['driver_vector']
        global_vector = self.global_graph(sub_vector.unsqueeze(0))
        output = torch.cat([driver_vector, global_vector], dim=1)
        # global_graph_input = veh_vector.unsqueeze(0)
        return output
