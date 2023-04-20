import numpy as np
import torch


def adms_collate(batch):
    polylines = np.concatenate([each[0] for each in batch])
    obj_num = np.array([each[0].shape[0] for each in batch])
    mask = np.concatenate([each[1] for each in batch])
    driver_dense_features = np.stack([each[2] for each in batch])
    labels = np.stack([each[3] for each in batch])
    labels = labels.reshape(-1, 1)
    veh_cate_features = torch.zeros(len(batch), 3).int()
    veh_dense_features = torch.zeros(len(batch), 4)
    driver_cate_features = torch.zeros(len(batch), 4).int()
    return {
        'inputs': {
            'veh_cate_features': veh_cate_features,
            'veh_dense_features': veh_dense_features,
            'driver_cate_features': driver_cate_features,
            'driver_dense_features': torch.from_numpy(driver_dense_features.astype(np.float32)),
            'polylines': torch.from_numpy(polylines.astype(np.float32)),
            'polynum': torch.from_numpy(obj_num.astype(np.int)),
            'attention_mask': torch.from_numpy(mask.astype(np.int))
        },
        'label': torch.from_numpy(labels)
    }


class ADMSDataset(torch.utils.data.Dataset):

    def __init__(self, file_dict):
        self.polylines = np.load(file_dict['polylines'], allow_pickle=True)
        self.masks = np.load(file_dict['masks'], allow_pickle=True)
        self.driver_dense_features = np.load(file_dict['driver_dense_features'], allow_pickle=True)
        self.labels = np.load(file_dict['labels'], allow_pickle=True)
        assert len(self.polylines) == len(self.labels)
        assert len(self.polylines) == len(self.masks)
        assert len(self.polylines) == len(self.driver_dense_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.polylines[idx], self.masks[idx], self.driver_dense_features[idx], self.labels[idx]
