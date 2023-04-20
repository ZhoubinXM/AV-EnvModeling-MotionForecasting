import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import time, json, datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DeepFM(nn.Module):

    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, emb_size=8, hid_dims=[256, 128], dropout=[0.2, 0.2]):
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况 
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size
        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, emb_size)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([nn.Embedding(voc_size, emb_size)
                                                      for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示

        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([nn.Embedding(voc_size, emb_size)
                                                      for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示
        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))
        # for output
        self.dnn_linear = nn.Linear(hid_dims[-1], emb_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """
        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size, emb_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1)  # [bs, emb_size]

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res  # [bs, 1]
        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]

        fm_2nd_part = sub  # [bs, emb_size]
        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, n * emb_size]

        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))  # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out  # [bs, n * emb_size]

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)  # [bs, emb_size]
        out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, emb_size]
        return out
