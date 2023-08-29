import torch
from torch import nn
from layer.base import MLP
import torch.nn.functional as F


class SubGraph(nn.Module):
    r"""
    Sub graph of VectorNet.

    It has three MLPs, each mlp is a fully connected layer followed by layer normalization and ReLU
    """

    def __init__(self, hidden_size, depth=3):
        super(SubGraph, self).__init__()
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, poly_num: torch.Tensor):
        if self.training:
            return self._train(hidden_states, attention_mask)
        else:
            poly_num = torch.tensor([hidden_states.shape[0]])
            # print(hidden_states.shape)
            # train_res = self._train(hidden_states, attention_mask)
            # inference_res = self._inference(hidden_states, attention_mask, poly_num)
            # if torch.sum(train_res - inference_res) != 0:
            #     print("error")
            # return self._train(hidden_states, attention_mask)
            return self._inference(hidden_states, attention_mask, poly_num)

    def _train(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        sub_graph_batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        hidden_size = hidden_states.shape[2]
        attention_mask = (attention_mask - 1) * 10000000.0
        for _, layer in enumerate(self.layers):
            encoded_hidden_states = layer(hidden_states)
            max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
            max_hidden = F.relu(max_hidden)
            max_hidden = torch.unsqueeze(max_hidden, 1)
            max_hidden = max_hidden.expand(
                [sub_graph_batch_size, max_vector_num,
                 torch.div(hidden_size, 2, rounding_mode='floor')])
            new_hidden_states = torch.cat((encoded_hidden_states, max_hidden), dim=-1)
            hidden_states = new_hidden_states
        sub_vec, _ = torch.max(hidden_states, dim=1)
        return sub_vec

    def _inference(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, poly_num: torch.Tensor):
        device = hidden_states.device
        index = torch.arange(0, poly_num[0], device=device)
        hidden_states = torch.index_select(hidden_states, dim=0, index=index)
        attention_mask = torch.index_select(attention_mask, dim=0, index=index)
        sub_graph_batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        hidden_size = hidden_states.shape[2]
        attention_mask = (attention_mask - 1) * 10000000.0
        for _, layer in enumerate(self.layers):
            encoded_hidden_states = layer(hidden_states)
            max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
            max_hidden = F.relu(max_hidden)
            max_hidden = torch.unsqueeze(max_hidden, 1)
            # max_hidden = max_hidden.expand(
            #     [sub_graph_batch_size, max_vector_num,
            #      torch.div(hidden_size, 2, rounding_mode='floor')])
            max_hidden = max_hidden.expand([sub_graph_batch_size, max_vector_num, int(hidden_size / 2)])
            new_hidden_states = torch.cat((encoded_hidden_states, max_hidden), dim=-1)
            hidden_states = new_hidden_states
        sub_vec, _ = torch.max(hidden_states, dim=1)
        return sub_vec
