from torch.utils.data import Dataset
import os
import cv2
import torch
import pickle
from numpy import random
from train_eval.utils import convert_list_to_array
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global, convert_global_coords_to_local


class PredictionDataset(Dataset):
    def __init__(self,
                 args,
                 mode='train',
                 use_aug=False,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.length = 0
        self.samples = []
        self.files = []
        self.mode = mode
        root_dir = args['root_dir']
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pkl'):
                    self.files.append(os.path.join(root, file))
        # Data Augmentation
        if self.mode == 'val':
            use_aug = False
        self.use_aug = use_aug
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            sample_token = f.name.split('/')[-1].split('_')[1].split('.')[0]
            instance_token = f.name.split('/')[-1].split('_')[0]
            data = pickle.load(f)
        del data['all_history']
        del data['all_category']
        del data['lanes']
        data['sample_token'] = sample_token
        data['instance_token'] = instance_token
        data['target_history'] = convert_global_coords_to_local(
            data['target_history'], data['target_translation'],
            data['target_rotation'])
        data['target_future'] = convert_global_coords_to_local(
            data['target_future'], data['target_translation'],
            data['target_rotation'])
        data = convert_list_to_array(data)
        if self.use_aug:
            img: torch.Tensor = data['img']
            img = img.permute(1, 2, 0).numpy(
            )  # Convert to numpy array and reshape to [244, 244, 3]
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta
            # Randomly select the mode for random contrast operation
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha
            # BGR -> HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Randomly change the saturation of the image
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)
            # Randomly change the hue of the image
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360
            # HSV -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            # Apply random contrast operation if not applied before
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha
            # Randomly swap the image channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            img = torch.tensor(img).permute(2, 0, 1)
            # Convert back to torch.Tensor and reshape to [3, 244, 244]
            data['img'] = img
        return data
