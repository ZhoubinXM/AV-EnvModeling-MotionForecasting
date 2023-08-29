import math
import torch
from torch import nn
import numpy as np

from train_eval.utils import show_heatmaps


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def masked_softmax_mask(X, mask, value=-1e6):
    if mask is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        X[~mask] = value
        return nn.functional.softmax(X, dim=-1)


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """
    def __init__(self, hidden_size: int, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    """PointWise FNN"""
    def __init__(self, hidden_size, out_features=None, bias=True):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features, bias=bias)
        self.layer_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        return hidden_states


class GlobalGraph(nn.Module):
    r"""
    Global graph

    It's actually a self-attention.
    """
    def __init__(self,
                 hidden_size,
                 attention_head_size=None,
                 num_attention_heads=1,
                 attention_decay=False):
        super(GlobalGraph, self).__init__()
        self.attention_decay = attention_decay
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        if attention_decay:
            self.attention_decay = nn.Parameter(torch.ones(1) * 0.5)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size),
            key_layer.transpose(-1, -2))
        # print(attention_scores.shape, attention_mask.shape)
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(
                attention_mask)
        # if utils.args.attention_decay and utils.second_span:
        #     attention_scores[:, 0, 0, 0] = attention_scores[:, 0, 0, 0] - self.attention_decay
        self.attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if mapping is not None:
            for i, each in enumerate(self.attention_probs.tolist()):
                mapping[i]['attention_scores'] = np.array(each[0])
        if self.attention_decay:
            # logging(self.attention_decay, prob=0.01)
            value_layer = torch.cat([
                value_layer[:, 0:1, 0:1, :] * self.attention_decay,
                value_layer[:, 0:1, 1:, :]
            ],
                                    dim=2)
        context_layer = torch.matmul(self.attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert self.attention_probs.shape[1] == 1
            self.attention_probs = torch.squeeze(self.attention_probs, dim=1)
            assert len(self.attention_probs.shape) == 3
            return context_layer, self.attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self,
                 hidden_size,
                 attention_head_size=None,
                 num_attention_heads=1,
                 key_hidden_size=None,
                 query_hidden_size=None):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size,
                                             num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size,
                                   self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size,
                                 self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size,
                                   self.all_head_size * self.num_qkv)

    def forward(self,
                hidden_states_query,
                hidden_states_key=None,
                attention_mask=None,
                mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size),
            key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] \
                   and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + self.get_extended_attention_mask(
                attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        # hidden_states = self.global_graph(hidden_states, attention_mask, mapping) \
        #                 + self.global_graph2(hidden_states, attention_mask, mapping)
        hidden_states = torch.cat([
            self.global_graph(hidden_states, attention_mask, mapping),
            self.global_graph2(hidden_states, attention_mask, mapping)
        ],
                                  dim=-1)
        return hidden_states


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        # scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        scores = torch.matmul(queries / math.sqrt(queries.shape[-1]),
                              keys.transpose(-1, -2))
        self.attention_weights = masked_softmax_mask(scores, mask)
        # return torch.bmm(self.dropout(self.attention_weights), values)
        return torch.matmul(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, mask):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # if valid_lens is not None:
        #     # On axis 0, copy the first item (scalar or vector) for
        #     # `num_heads` times, then copy the next item, and so on
        #     valid_lens = torch.repeat_interleave(
        #         valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, mask)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = output.permute(0, 2, 1, 3)
        output_concat = output_concat.reshape(output_concat.shape[0],
                                              output_concat.shape[1], -1)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    # return X.reshape(-1, X.shape[2], X.shape[3])
    return X


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer encoder block.

    Defined in :numref:`sec_transformer`"""
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 dropout,
                 use_bias=False,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout,
                                            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, mask):
        Y = self.addnorm1(X, self.attention(X, X, X, mask))
        return self.addnorm2(Y, self.ffn(Y))
    

from metric.metric import Metric
from typing import Dict, Union
import torch
from metric.utils import min_ade, traj_nll


class MTPLoss(Metric):
    """
    MTP loss modified to include variances. Uses MSE for mode selection. Can also be used with
    Multipath outputs, with residuals added to anchors.
    """

    def __init__(self, args: Dict = None):
        """
        Initialize MTP loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component, compared to regression component
                of loss, default: 1
        """
        self.use_variance = args['use_variance'] if args is not None and 'use_variance' in args.keys() else False
        self.alpha = args['alpha'] if args is not None and 'alpha' in args.keys() else 1
        self.beta = args['beta'] if args is not None and 'beta' in args.keys() else 1
        self.name = 'mtp_loss'

    def __call__(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute MTP loss
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode (log) probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """

        # Unpack arguments
        traj = predictions['traj']
        log_probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 5 if self.use_variance else 2

        # Masks for variable length ground truth trajectories
        masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)

        # Obtain mode with minimum ADE with respect to ground truth:
        errs, inds = min_ade(traj, traj_gt, masks)
        inds_rep = inds.repeat(sequence_length, pred_params, 1, 1).permute(3, 2, 0, 1)

        # Calculate MSE or NLL loss for trajectories corresponding to selected outputs:
        traj_best = traj.gather(1, inds_rep).squeeze(dim=1)

        if self.use_variance:
            l_reg = traj_nll(traj_best, traj_gt, masks)
        else:
            l_reg = errs

        # Compute classification loss
        l_class = - torch.squeeze(log_probs.gather(1, inds.unsqueeze(1)))

        loss = self.beta * l_reg + self.alpha * l_class
        loss = torch.mean(loss)

        return loss