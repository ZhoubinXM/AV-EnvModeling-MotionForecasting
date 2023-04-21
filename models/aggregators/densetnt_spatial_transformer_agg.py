import torch
from torch import nn

from layer.base import GlobalGraph, GlobalGraphRes
from train_eval.utils import merge_tensors, get_from_mapping
from typing import List, Dict, Optional


class SpatialTransformerEncoder(nn.Module):
    """Spatial Encoder to get Spatial Attention feature"""
    def __init__(self, args: Dict) -> None:
        super().__init__()
        hidden_size = args['hidden_size']
        attention_head_size = None
        num_attention_heads = 1
        attention_decay = False
        # Use Multi-head attention block
        self.global_graph = GlobalGraph(hidden_size)

        self.global_graph_res = GlobalGraphRes(hidden_size)

    def forward(self,
                mapping: List[Dict],
                device: Optional[int] = 0) -> torch.Tensor:
        element_states_batch = get_from_mapping(mapping, 'env_seq_encode')
        batch_size = len(mapping)
        inputs, inputs_lengths = merge_tensors(element_states_batch,
                                               device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num],
                                     device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        return hidden_states
