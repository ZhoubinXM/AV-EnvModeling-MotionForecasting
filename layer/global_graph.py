import torch
import torch.nn.functional as F
from torch import nn
import math


class GlobalGraph(nn.Module):
    r"""
    Global graph

    It's actually a self-attention.
    """

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        # attention_mask shape: (batch, max_poly_num, max_poly_num)
        extended_attention_mask = attention_mask.unsqueeze(1)  #(batch, 1, max_poly_num, max_poly_num)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # (batch, max_poly_num, head_num, head_size)
        x = x.view(*sz)
        # (batch, head_num, max_poly_num, head_size)
        return x.permute(0, 2, 1, 3)
     

    def forward(self, hidden_states, attention_mask=None):
        if self.training:
            assert attention_mask is not None
            return self._train(hidden_states, attention_mask)
        else:
            # inference_res = self._inference(hidden_states)
            # train_res = self._train(hidden_states, attention_mask)
            # if torch.sum(inference_res - train_res) != 0:
            #     print("error")
            # return self._train(hidden_states, attention_mask)
            return self._inference(hidden_states)

    def _train(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer / self.sqrt_attention_head_size, key_layer.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # (batch, head_num, max_poly_num, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch, max_poly_num, head_num, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch, max_poly_num, head_num * head_size)
        return torch.max(context_layer, dim=1)[0]
        # return context_layer[:, 0, :]

    # for inference: batch size = 1 and attention_mask is not necessary
    def _inference(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # attention_scores = torch.matmul(query_layer / self.sqrt_attention_head_size, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # (batch, head_num, poly_num, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch, poly_num, head_num, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch, poly_num, head_num * head_size)
        return torch.max(context_layer, dim=1)[0]

        # return context_layer[:, 0, :]
