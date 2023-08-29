import numpy as np
import torch
from torch import nn, tensor, Tensor
from torch.nn import functional as F

from typing import Dict, List, Union, Optional

from layer.base import MLP, GlobalGraph, EncoderBlock
from train_eval.utils import get_from_mapping, get_max_st_from_spans


class WayFormerEncoder(nn.Module):
    """WayFormer Encoder"""
    def __init__(self, args: Dict) -> None:
        super().__init__()
        # Init encoder args.
        hidden_size = args['hidden_size']
        num_layers = args['num_layers']
        fusion_type = args['fusion_type']
        attention_type = args['attention_type']
        block_stack_type = args['block_stack_type']
        self.max_temporal = args['max_temporal']
        self.max_spatial = args['max_spatial']
        self.num_heads = args['num_heads']
        self.dropout = args['dropout']
        self.hidden_size = hidden_size
        max_len = self.max_temporal

        self.projection_layer = MLP(hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(
            (1, self.max_temporal, hidden_size)),
                                          requires_grad=True)
        if fusion_type == 'late_fusion':
            if attention_type == 'factorized_attention':
                self.fusion_layer = nn.Sequential()
                for i in range(num_layers):
                    if block_stack_type == 'sequential':
                        if i == num_layers // 2:
                            max_len = self.max_spatial
                        self.fusion_layer.add_module(
                            f"{i}",
                            EncoderBlock(self.hidden_size,
                                         self.hidden_size,
                                         self.hidden_size,
                                         self.hidden_size,
                                         [self.hidden_size],
                                         self.hidden_size,
                                         self.hidden_size*2,
                                         num_heads=self.num_heads,
                                         dropout=self.dropout))
                    else:
                        raise NotImplementedError(
                            f"{block_stack_type} is Not Implemented")
            else:
                raise NotImplementedError(
                    f"{attention_type} is Not Implemented!")
        else:
            raise NotImplementedError(f"{fusion_type} is Not Implemented!")

    def forward(self, mapping: List[Dict], device: Optional[int] = 0):
        """WayFormer Encoder forward main function"""

        input_list_list: List[List[torch.Tensor]] = []
        lane_states_batch = None
        batch_size = len(mapping)
        polyline_spans = get_from_mapping(mapping, 'polyline_spans')
        polyline_matrixs = get_from_mapping(mapping, 'matrix')

        # get valid length
        spatial_num, slice_num_list = get_max_st_from_spans(polyline_spans)
        # max_spatial_num = max(spatial_num)
        # max_vector_num = max(max(row) for row in slice_num_list)
        mask_spatial_vector = np.zeros([
            batch_size, self.max_spatial, self.max_temporal, self.hidden_size
        ])
        mask_temporal = np.ones([
            batch_size, self.max_spatial, self.max_temporal, self.max_temporal
        ])
        mask_spatial = np.ones([
            batch_size, self.max_temporal, self.max_spatial, self.max_spatial
        ])
        for i_data in range(batch_size):
            input_list: List[np.ndarray] = []
            for poly_idx, polyline_span in enumerate(polyline_spans[i_data]):
                poly_tensor: np.ndarray = np.pad(
                    polyline_matrixs[i_data][polyline_span],
                    ((0, self.max_temporal - slice_num_list[i_data][poly_idx]),
                     (0, 0)),
                    'constant',
                    constant_values=(-1, -1))
                mask_spatial_vector[i_data][poly_idx][:(
                    slice_num_list[i_data][poly_idx]), :] = 1
                input_list.append(poly_tensor)
            slice_num_list[i_data] += [0] * (self.max_spatial -
                                             spatial_num[i_data])
            input_list = np.pad(input_list,
                                ((0, self.max_spatial - len(input_list)),
                                 (0, 0), (0, 0)),
                                'constant',
                                constant_values=((-1, -1), (-1, -1), (-1, -1)))
            input_list_list.append(input_list)
        # Shape[B, S, T, D]
        all_input_tensor = torch.tensor(np.array(input_list_list),
                                        dtype=torch.float,
                                        device=device)
        # get each mask
        # for B
        for i in range(mask_spatial_vector.shape[0]):
            batch_mask = mask_spatial_vector[i]
            # for S
            for j in range(batch_mask.shape[0]):
                spatial_mask = batch_mask[j]
                # for T
                for k in range(spatial_mask.shape[0]):
                    if spatial_mask[k][0] == 0:
                        mask_temporal[i][j][k, :] = 0
                        mask_temporal[i][j][:, k] = 0

            # for T
            for l in range(batch_mask.shape[1]):
                temproal_mask = batch_mask[:, l, :]
                # for S
                for m in range(temproal_mask.shape[0]):
                    if temproal_mask[m][0] == 0:
                        mask_spatial[i][l][m, :] = 0
                        mask_spatial[i][l][:, m] = 0

        # Shape[B*S, T, D]
        hidden_state = all_input_tensor.reshape(-1, all_input_tensor.shape[2],
                                                all_input_tensor.shape[3])
        mask = torch.tensor(mask_temporal, device=device).bool()
        hidden_state = self.projection_layer(hidden_state)
        hidden_state = hidden_state + self.pos_embedding.data[:, :hidden_state.
                                                              shape[1], :]
        mask = mask.reshape(-1, mask.shape[2],
                            mask.shape[3]).unsqueeze(1).repeat(
                                1, self.num_heads, 1, 1)
        for layer_idx, block in enumerate(self.fusion_layer):
            if layer_idx == len(self.fusion_layer) // 2:
                # hidden_state shape[B*S, T, D] -> [B*T, S, D]
                mask = torch.tensor(mask_spatial, device=device).bool()
                mask = mask.reshape(-1, mask.shape[2],
                                    mask.shape[3]).unsqueeze(1).repeat(
                                        1, self.num_heads, 1, 1)
                hidden_state = hidden_state.reshape(batch_size,
                                                    self.max_spatial,
                                                    self.max_temporal,
                                                    self.hidden_size)
                hidden_state = hidden_state.permute(0, 2, 1, 3)
                hidden_state = hidden_state.reshape(-1, self.max_spatial,
                                                    self.hidden_size)

            hidden_state: torch.Tensor = block(hidden_state, mask)

        return hidden_state.reshape(batch_size, self.max_spatial,
                                    self.max_temporal, self.hidden_size)[:,
                                                                         0, :]
