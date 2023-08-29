from torch.utils.data import Dataset
import os
import cv2
import torch
import pickle
import numpy as np
from numpy import random
from train_eval.utils import convert_list_to_array
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global, convert_global_coords_to_local

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from data_process.create_anchor import classify_label_type_to_id


class MotionDataset(Dataset):
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
        #
        # self.nusc = NuScenes(version='v1.0-trainval',
        #                      dataroot="./data/nuscenes/trainval",
        #                      verbose=True)
        # self.pred_helper = PredictHelper(nusc=self.nusc)
        # self.predict_steps = 12
        # self.past_steps = 6

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            sample_token = f.name.split('/')[-1].split('_')[1].split('.')[0]
            instance_token = f.name.split('/')[-1].split('_')[0]
            data = pickle.load(f)
        # data['sample_token'] = sample_token
        # data['instance_token'] = instance_token
        # data['target_history'] = convert_global_coords_to_local(
        #     data['target_history'], data['ego_translation'],
        #     data['ego_rotation'])
        # data['target_future'] = convert_global_coords_to_local(
        #     data['target_future'], data['ego_translation'],
        #     data['ego_rotation'])
        # sdc_fut_traj_all, sdc_fut_traj_valid_mask_all, \
        # sdc_past_traj_all, sdc_past_traj_valid_mask_all = \
        #   self.get_sdc_traj_label(sample_token)
        # all_history = data['all_history']
        # agent_past_feature = []
        # agent_past_feature_mask = []
        # agent_cat = []
        # agent_fut_feature = []
        # agent_fut_feature_mask = []
        # for ins, cat in data['all_category'].items():
        #     if ins == instance_token:
        #         agent_past_feature.append(
        #             self.get_full_array(data['target_history']))
        #         agent_cat.append(classify_label_type_to_id(cat))
        #         agent_past_feature_mask.append(
        #             self.get_valid_mask(data['target_history']))
        #         agent_fut_feature.append(
        #             self.get_full_array(data['target_future'], step=12))
        #         agent_fut_feature_mask.append(
        #             self.get_valid_mask(data['target_future'], 12))
        #         continue
        #     agent_past_feature.append(
        #         self.get_full_array(convert_global_coords_to_local(all_history[ins],
        #                                        data['ego_translation'],
        #                                        data['ego_rotation'])))
        #     agent_cat.append(classify_label_type_to_id(cat))

        #     agent_past_feature_mask.append(
        #         self.get_valid_mask(all_history[ins]))
        #     agent_fut = self.pred_helper.get_future_for_agent(
        #         ins, sample_token, seconds=6, in_agent_frame=False)
        #     if agent_fut.size > 0:
        #         agent_fut = convert_global_coords_to_local(
        #             agent_fut, data['ego_translation'], data['ego_rotation'])
        #     agent_fut_feature.append(self.get_full_array(agent_fut, 12))
        #     agent_fut_feature_mask.append(self.get_valid_mask(agent_fut, 12))

        # data['agent_past_feature'] = agent_past_feature
        # data['agent_category'] = agent_cat
        # data['agent_past_feature_mask'] = agent_past_feature_mask
        # data['agent_fut_feature'] = agent_fut_feature
        # data['agent_fut_feature_mask'] = agent_fut_feature_mask
        # data['sdc_fut_traj_all'] = sdc_fut_traj_all
        # data['sdc_fut_traj_valid_mask_all'] = sdc_fut_traj_valid_mask_all
        # data['sdc_past_traj_all'] = sdc_past_traj_all
        # data['sdc_past_traj_valid_mask_all'] = sdc_past_traj_valid_mask_all
        # del data['all_history']
        # del data['all_category']
        # del data['lanes']
        # data = convert_list_to_array(data)
        # if self.use_aug:
        #     img: torch.Tensor = data['img']
        #     img = img.permute(1, 2, 0).numpy(
        #     )  # Convert to numpy array and reshape to [244, 244, 3]
        #     # random brightness
        #     if random.randint(2):
        #         delta = random.uniform(-self.brightness_delta,
        #                                self.brightness_delta)
        #         img += delta
        #     # Randomly select the mode for random contrast operation
        #     mode = random.randint(2)
        #     if mode == 1:
        #         if random.randint(2):
        #             alpha = random.uniform(self.contrast_lower,
        #                                    self.contrast_upper)
        #             img *= alpha
        #     # BGR -> HSV
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #     # Randomly change the saturation of the image
        #     if random.randint(2):
        #         img[..., 1] *= random.uniform(self.saturation_lower,
        #                                       self.saturation_upper)
        #     # Randomly change the hue of the image
        #     if random.randint(2):
        #         img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
        #         img[..., 0][img[..., 0] > 360] -= 360
        #         img[..., 0][img[..., 0] < 0] += 360
        #     # HSV -> BGR
        #     img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        #     # Apply random contrast operation if not applied before
        #     if mode == 0:
        #         if random.randint(2):
        #             alpha = random.uniform(self.contrast_lower,
        #                                    self.contrast_upper)
        #             img *= alpha
        #     # Randomly swap the image channels
        #     if random.randint(2):
        #         img = img[..., random.permutation(3)]
        #     img = torch.tensor(img).permute(2, 0, 1)
        #     # Convert back to torch.Tensor and reshape to [3, 244, 244]
        #     data['img'] = img
        
        # save_path = '/data/jerome.zhou/nuscenes/val/'
        # # os.makedirs(save_path)
        # file_name = f'{instance_token}_{sample_token}.pkl'
        # with open(save_path+file_name, 'wb') as f:
        #     pickle.dump(data, f)
        return data

    def get_sdc_traj_label(self, sample_token):
        sd_rec = self.nusc.get('sample', sample_token)
        sd_rec_ori = sd_rec
        lidar_top_data_start = self.nusc.get('sample_data',
                                             sd_rec['data']['LIDAR_TOP'])
        ego_pose_start = self.nusc.get('ego_pose',
                                       lidar_top_data_start['ego_pose_token'])

        sdc_fut_traj = []
        for _ in range(self.predict_steps):
            next_annotation_token = sd_rec['next']
            if next_annotation_token == '':
                break
            sd_rec = self.nusc.get('sample', next_annotation_token)
            lidar_top_data_next = self.nusc.get('sample_data',
                                                sd_rec['data']['LIDAR_TOP'])
            ego_pose_next = self.nusc.get(
                'ego_pose', lidar_top_data_next['ego_pose_token'])
            sdc_fut_traj.append(ego_pose_next['translation']
                                [:2])  # global xy pos of sdc at future step i

        sdc_fut_traj_all = np.zeros((self.predict_steps, 2))
        sdc_fut_traj_valid_mask_all = np.zeros((self.predict_steps, 2))
        n_valid_timestep = len(sdc_fut_traj)
        if n_valid_timestep > 0:
            sdc_fut_traj = np.stack(sdc_fut_traj, axis=0)  #(t,2)
            sdc_fut_traj = convert_global_coords_to_local(
                coordinates=sdc_fut_traj,
                translation=ego_pose_start['translation'],
                rotation=ego_pose_start['rotation'],
            )
            sdc_fut_traj_all[:n_valid_timestep, :] = sdc_fut_traj
            sdc_fut_traj_valid_mask_all[:n_valid_timestep, :] = 1
        sdc_past_traj = []
        sdc_past_traj.append(ego_pose_start['translation'][:2])
        sd_rec = sd_rec_ori
        for _ in range(self.past_steps):
            next_annotation_token = sd_rec['prev']
            if next_annotation_token == '':
                break
            sd_rec = self.nusc.get('sample', next_annotation_token)
            lidar_top_data_next = self.nusc.get('sample_data',
                                                sd_rec['data']['LIDAR_TOP'])
            ego_pose_next = self.nusc.get(
                'ego_pose', lidar_top_data_next['ego_pose_token'])
            sdc_past_traj.append(ego_pose_next['translation']
                                 [:2])  # global xy pos of sdc at future step i

        sdc_past_traj_all = np.zeros((self.past_steps + 1, 2))
        sdc_past_traj_valid_mask_all = np.zeros((self.past_steps + 1, 2))
        n_valid_timestep_past = len(sdc_past_traj)
        if n_valid_timestep_past > 0:
            sdc_past_traj = np.stack(sdc_past_traj, axis=0)  #(t,2)
            sdc_past_traj = convert_global_coords_to_local(
                coordinates=sdc_past_traj,
                translation=ego_pose_start['translation'],
                rotation=ego_pose_start['rotation'],
            )
            sdc_past_traj_all[:n_valid_timestep_past, :] = sdc_past_traj
            sdc_past_traj_valid_mask_all[:n_valid_timestep_past, :] = 1

        return sdc_fut_traj_all, sdc_fut_traj_valid_mask_all, sdc_past_traj_all, sdc_past_traj_valid_mask_all

    def get_valid_mask(self, trajs, step=7):
        # trajs_all = np.zeros((step, 2))
        trajs_mask_all = np.zeros((step, 2))
        n_valid_timestamp = len(trajs)
        if n_valid_timestamp > 0:
            # trajs = np.stack(trajs, axis=0)
            trajs_mask_all[:n_valid_timestamp, :] = 1

        return trajs_mask_all

    def get_full_array(self, trajs, step=7):
        trajs_all = np.zeros((step, 2))
        n_valid_timestamp = len(trajs)
        if n_valid_timestamp > 0:
            # trajs = np.stack(trajs, axis=0)
            trajs_all[:n_valid_timestamp, :] = trajs

        return trajs_all
    