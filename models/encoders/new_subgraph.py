import torch
from torch import nn, Tensor
from torch.nn import functional as F

from layer.base import MLP, LayerNorm, GlobalGraph, CrossAttention, GlobalGraphRes
from typing import Dict, Optional, List
from train_eval.utils import get_from_mapping, merge_tensors, de_merge_tensors, get_max_st_from_spans
import numpy as np

class NewSubGraph(nn.Module):
    def __init__(self, args: Dict):
        super(NewSubGraph, self).__init__()
        hidden_size = args['hidden_size']
        depth = args['depth']
        if depth is None:
            depth = 3
        self.layer_0 = MLP(hidden_size)
        self.layer_0_again = MLP(hidden_size)
        self.layers = nn.ModuleList([
            GlobalGraph(hidden_size, num_attention_heads=2)
            for _ in range(depth)
        ])
        self.layers_2 = nn.ModuleList(
            [LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_3 = nn.ModuleList(
            [LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_4 = nn.ModuleList(
            [GlobalGraph(hidden_size) for _ in range(depth)])

        # LaneGCN
        self.laneGCN_A2L = CrossAttention(hidden_size)
        self.laneGCN_L2L = GlobalGraphRes(hidden_size)
        self.laneGCN_L2A = CrossAttention(hidden_size)

    def temporal_encoder(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = merge_tensors(input_list, device)
        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device)
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(
            de_merge_tensors(hidden_states, lengths))

    def forward(self, mapping: List[Dict], device: Optional[int] = 0) -> List[Dict]:
        """Encoder forward main function"""
        input_list_list: List[List[torch.Tensor]] = []
        map_input_list_list: List[List[torch.Tensor]] = []
        lane_states_batch = None
        batch_size = len(mapping)
        polyline_spans = get_from_mapping(mapping, 'polyline_spans')
        polyline_matrixs = get_from_mapping(mapping, 'matrix')

        self.hidden_size = 128
        spatial_num, slice_num_list = get_max_st_from_spans(polyline_spans)
        max_spatial_num = max(spatial_num)
        max_vector_num = max(max(row) for row in slice_num_list)
        mask_spatial_vector = np.zeros([batch_size, max_spatial_num, max_vector_num, self.hidden_size])
        for i_data in range(batch_size):
            input_list: List[np.ndarray] = []
            for poly_idx, polyline_span in enumerate(polyline_spans[i_data]):
                poly_tensor:np.ndarray = np.pad(
                    polyline_matrixs[i_data][polyline_span],
                    ((0, max_vector_num - slice_num_list[i_data][poly_idx]),
                     (0, 0)),
                    'constant',
                    constant_values=(-1, -1))
                mask_spatial_vector[i_data][poly_idx][:(slice_num_list[i_data][poly_idx]), :] = 1.
                input_list.append(poly_tensor)
            slice_num_list[i_data] += [0]*(max_spatial_num - spatial_num[i_data])
            input_list = np.pad(input_list,
                                ((0, max_spatial_num - len(input_list)), (0, 0), (0, 0)),
                                'constant',
                                constant_values=((-1, -1), (-1, -1), (-1, -1)))
            input_list_list.append(input_list)
        # Shape[B, S, T, D]
        all_input_tensor = torch.tensor(input_list_list, dtype=torch.float, device=device)


        for i_data in range(batch_size):
            input_list: List[torch.Tensor] = []
            map_input_list: List[torch.Tensor] = []
            map_polyline_start_idx: int = mapping[i_data][
                'map_start_polyline_idx']
            for poly_idx, polyline_span in enumerate(polyline_spans[i_data]):
                poly_tensor = torch.tensor(
                    polyline_matrixs[i_data][polyline_span], device=device)
                input_list.append(poly_tensor)
                if poly_idx > map_polyline_start_idx:
                    map_input_list.append(poly_tensor)
            input_list_list.append(input_list)
            map_input_list_list.append(map_input_list)

        # input_list_list [B, S_all(dynamic), T(dynamic), D]
        # map_input_lis_list [B, S_map(dynamic), T(dynamic), D]
        # Process: Temporal Transformer encoder
        element_states_batch: List[torch.Tensor] = []
        for i in range(batch_size):
            max_seq_feature, _ = self.temporal_encoder(input_list_list[i])
            element_states_batch.append(max_seq_feature)

        # Process lane: Temporal Transformer encoder
        lane_states_batch = []
        for i in range(batch_size):
            max_seq_feature, _ = self.temporal_encoder(map_input_list_list[i])
            lane_states_batch.append(max_seq_feature)

        # Get agent and lane feature do LaneGCN encoder
        for i in range(batch_size):
            map_polyline_start_idx: int = mapping[i_data][
                'map_start_polyline_idx']
            agents_feature = element_states_batch[i][:map_polyline_start_idx]
            lanes_feature = element_states_batch[i][map_polyline_start_idx:]
            lanes_feature = lanes_feature + self.laneGCN_A2L(
                lanes_feature.unsqueeze(0),
                torch.cat([lanes_feature, agents_feature[0:1]
                           ]).unsqueeze(0)).squeeze(0)
            # lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), agents.unsqueeze(0)).squeeze(0)
            # lanes = lanes + self.laneGCN_L2L(lanes.unsqueeze(0)).squeeze(0)
            # agents = agents + self.laneGCN_L2A(agents.unsqueeze(0), lanes.unsqueeze(0)).squeeze(0)
            element_states_batch[i] = torch.cat(
                [agents_feature, lanes_feature])

        # Update encoder feature in mapping
        for i in range(batch_size):
            mapping[i].update(
                dict(
                    env_seq_encode=element_states_batch[i],
                    lane_seq_encode=lane_states_batch[i],
                ))

        return mapping
