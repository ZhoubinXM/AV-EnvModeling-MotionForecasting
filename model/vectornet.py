import torch
import torch.nn.functional as F
from torch import nn
import math
from model.tnt_s1s2s3 import TNTDecoder


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        return hidden_states


class SubGraph(nn.Module):
    r"""
    Sub graph of VectorNet.

    It has three MLPs, each mlp is a fully connected layer followed by layer normalization and ReLU
    """
    def __init__(self, hidden_size, depth=3):
        super(SubGraph, self).__init__()
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        sub_graph_batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        hidden_size = hidden_states.shape[2]
        for _, layer in enumerate(self.layers):
            encoded_hidden_states = layer(hidden_states)
            max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
            max_hidden = F.relu(max_hidden)
            max_hidden = torch.unsqueeze(max_hidden, 1)
            max_hidden = max_hidden.expand([sub_graph_batch_size, max_vector_num, hidden_size // 2])
            new_hidden_states = torch.cat((encoded_hidden_states, max_hidden), dim=-1)
            hidden_states = new_hidden_states
        sub_vec, _ = torch.max(hidden_states, dim=1)
        return sub_vec


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

    def transpose_for_scores(self, x, is_train=True):
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # (batch, max_poly_num, head_num, head_size)
        x = x.view(*sz)
        # (batch, head_num, max_poly_num, head_size)
        if is_train:
            return x.permute(0, 2, 1, 3)
        else:
            return x.permute(1, 0, 2)

    def forward(self, hidden_states, attention_mask=None, is_train=True):
        if is_train:
            assert attention_mask is not None
            return self._train(hidden_states, attention_mask)
        else:
            return self._inference(hidden_states)

    def _train(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # (batch, head_num, max_poly_num, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch, max_poly_num, head_num, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch, max_poly_num, head_num * head_size)
        return context_layer[:, 0, :]

    # for inference: batch size = 1 and attention_mask is not necessary
    def _inference(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # mix_hidden = torch.unsqueeze(hidden_states, 1)
        mixed_value_layer = self.value(hidden_states)
        # mixed_value_layer = torch.squeeze(mixed_value_layer, 1)

        query_layer = self.transpose_for_scores(mixed_query_layer, is_train=False)
        key_layer = self.transpose_for_scores(mixed_key_layer, is_train=False)
        value_layer = self.transpose_for_scores(mixed_value_layer, is_train=False)

        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)  # (head_num, poly_num, head_size)
        context_layer = context_layer.permute(1, 0, 2).contiguous()  # (poly_num, head_num, head_size)
        context_layer = context_layer.view(context_layer.size()[0], self.all_head_size)  # (poly_num, head_num * head_size)
        # return context_layer[0, :]
        # print(hidden_states.shape)
        # return hidden_states[1,:]
        return mixed_value_layer[0, :]


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """
    def __init__(self, hidden_size=128, mode="train", device=torch.device("cpu")):
        super(VectorNet, self).__init__()
        self.sub_graph = SubGraph(hidden_size)
        self.global_graph = GlobalGraph(hidden_size)
        self.decoder = TNTDecoder(in_channels=hidden_size, hidden_dim=64, N=2000, M=50)
        self.hidden_size = hidden_size
        assert mode in ["train", "inference"], "mode must be train or inference"
        self.mode = mode
        self.device = device

    def set_mode(self, mode):
        assert mode in ["train", "inference"], "mode must be train or inference"
        self.mode = mode

    def forward(self, polylines, poly_num, attention_mask, tar_candidate, tar_candidate_mask, gt_loss_param=None):
        if self.mode == "train":
            return self._train(polylines, poly_num, attention_mask, tar_candidate, tar_candidate_mask, gt_loss_param)
        else:
            return self._inference(polylines, poly_num, attention_mask, tar_candidate, tar_candidate_mask)

    def _train(self, polylines, poly_num, attention_mask, tar_candidate, tar_candidate_mask, gt_loss_param):
        device = self.device
        sub_vector = self.sub_graph(polylines, attention_mask)
        global_graph_batch = poly_num.shape[0]
        max_poly_num = torch.max(poly_num)
        global_graph_input = torch.zeros([global_graph_batch, max_poly_num, self.hidden_size], device=device)
        global_graph_attention_mask = torch.zeros([global_graph_batch, max_poly_num, max_poly_num], device=device)
        idx = 0
        for i, length in enumerate(poly_num):
            global_graph_input[i, :length, :] = sub_vector[idx:idx + length]
            global_graph_attention_mask[i, :length, :length] = torch.ones([length, length], device=device)
            idx += length
        global_vector = self.global_graph(global_graph_input, global_graph_attention_mask)
        return self.decoder(global_vector, tar_candidate, tar_candidate_mask, gt_loss_param, is_train=True)

    # for inference: batch size = 1
    def _inference(self, polylines, poly_num, attention_mask, tar_candidate, tar_candidate_mask):
        # **shape** batch size = 1
        # polylines: [100, 19, self.hidden_size]
        # poly_num: [1,]
        # attention_mask: [100, 19, self.hidden_size // 2]
        # global_vector: [1, self.hidden_size]
        polylines = polylines[:poly_num[0], :, :]
        attention_mask = attention_mask[:poly_num[0], :, :]
        sub_vector = self.sub_graph(polylines, attention_mask)
        global_vector = self.global_graph(sub_vector, is_train=False)
        pred_traj, prob = self.decoder(global_vector, tar_candidate, tar_candidate_mask, is_train=False)
        return pred_traj, prob
