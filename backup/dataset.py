from cProfile import label
import math
import multiprocessing
import os
import pickle
import zlib
from multiprocessing import Process

import numpy as np
import torch
from tqdm import tqdm
import sys
import copy


def collate_function(batch):
    polylines = np.concatenate([each[0] for each in batch])
    obj_num = np.array([each[0].shape[0] for each in batch])
    mask = np.concatenate([each[1] for each in batch])
    driver_dense_features = np.stack([each[2] for each in batch])
    labels = np.stack([each[3] for each in batch])
    return torch.from_numpy(polylines).float(), torch.from_numpy(obj_num).int(), torch.from_numpy(mask).int(), torch.from_numpy(
        driver_dense_features).float(), torch.from_numpy(labels).float()


class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        # self.objs = np.load("/home/guifeng.fan/project/adms/adms-model/processed_data/obj_vector_all.npy", allow_pickle=True)
        # self.masks = np.load("/home/guifeng.fan/project/adms/adms-model/processed_data/mask_all.npy", allow_pickle=True)
        # self.labels = np.load("/home/guifeng.fan/project/adms/adms-model/processed_data/label_all.npy", allow_pickle=True)
        # self.driver_dense_features = np.load(
        #     "/home/guifeng.fan/project/adms/adms-model/processed_data/driver_dense_feature_all.npy", allow_pickle=True)
        self.objs = np.load("/home/guifeng.fan/project/adms/adms-model/processed_data/obj_vector_1219.npy", allow_pickle=True)
        self.masks = np.load("/home/guifeng.fan/project/adms/adms-model/processed_data/mask_1219.npy", allow_pickle=True)
        self.labels = np.load("/home/guifeng.fan/project/adms/adms-model/processed_data/label_1219.npy", allow_pickle=True)
        self.driver_dense_features = np.load(
            "/home/guifeng.fan/project/adms/adms-model/processed_data/driver_dense_feature_1.npy", allow_pickle=True)
        assert len(self.objs) == len(self.labels)
        assert len(self.objs) == len(self.masks)
        assert len(self.objs) == len(self.driver_dense_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.objs[idx], self.masks[idx], self.driver_dense_features[idx], self.labels[idx]


if __name__ == "__main__":
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for idx, data in enumerate(dataloader):
        print(idx, data[0].shape, data[1].shape, data[2].shape)
        # print(data[2])
        # break
