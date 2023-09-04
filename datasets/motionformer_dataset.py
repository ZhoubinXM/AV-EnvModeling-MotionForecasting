from torch.utils.data import Dataset
import os
import cv2
import torch
import copy
import pickle
import numpy as np
from numpy import random
import sys

sys.path.append("..")
sys.path.append(".")
from train_eval.utils import convert_list_to_array, load_anchors
from nuscenes.prediction import PredictHelper, make_2d_rotation_matrix, convert_global_coords_to_local

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from data_process.create_anchor import classify_label_type_to_id

from visualize.camera_render import CameraRender, CAM_NAMES
from visualize.bev_render import BEVRender


class MotionFormerDataset(Dataset):
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
        # root_dir = args['root_dir']
        self.ann_file = args['root_dir']

        self.load_from_disk = args['load_from_disk']
        self.filter_velocity_before_train = args['filter_velocity']
        self.data_infos = self.load_annotation_file()
        # Data Augmentation
        if self.mode == 'val':
            use_aug = False
        self.use_aug = use_aug
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.debug = False
        self.predict_steps = 12
        self.past_steps = 6
        # only predict objs in this range.
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if not self.load_from_disk or self.debug:
            print("[INFO]: Loading pre-process anchors ...")
            self.anchors = load_anchors()
            dataroot = "./data/nuscenes/trainval"
            self.nusc = NuScenes(version='v1.0-mini',
                                 dataroot="./data/nuscenes/trainval",
                                 verbose=True)
            self.pred_helper = PredictHelper(nusc=self.nusc)

            from nuscenes.map_expansion.map_api import NuScenesMap
            self.nusc_maps = {
                'boston-seaport':
                NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
                'singapore-hollandvillage':
                NuScenesMap(dataroot=dataroot,
                            map_name='singapore-hollandvillage'),
                'singapore-onenorth':
                NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
                'singapore-queenstown':
                NuScenesMap(dataroot=dataroot,
                            map_name='singapore-queenstown'),
            }

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.load_from_disk:
            data = copy.deepcopy(self.data_infos[idx])
            sample_token = data['frame_token']
            ego_trans = np.array(data['ego2global_translation'])
            ego_yaw = data['ego2global_yaw']
            sdc_fut_traj = data['sdc_fut_traj']
            sdc_fut_traj_valid_mask = data['sdc_fut_traj_valid_mask']
            sdc_past_traj = data['sdc_past_traj']
            sdc_past_traj_valid_mask = data['sdc_past_traj_valid_mask']

            # no fut trajs or past trajs, reindex
            if np.all(data['fut_trajs_valid_mask'] == 0) or \
                np.all(data['past_trajs_valid_mask'] == 0):
                reindex = np.random.randint(0, len(self))
                return self[reindex]

            # add sdc feature to frame.
            data['frame_past_traj'] = np.concatenate(
                (np.expand_dims(sdc_past_traj, axis=0), data['past_trajs']),
                axis=0)
            data['frame_past_traj_valid_mask'] = np.concatenate(
                (np.expand_dims(sdc_past_traj_valid_mask,
                                axis=0), data['past_trajs_valid_mask']),
                axis=0)
            data['frame_fut_traj'] = np.concatenate(
                (np.expand_dims(sdc_fut_traj, axis=0), data['fut_trajs']),
                axis=0)
            data['frame_fut_traj_valid_mask'] = np.concatenate(
                (np.expand_dims(sdc_fut_traj_valid_mask,
                                axis=0), data['fut_trajs_valid_mask']),
                axis=0)
            data['frame_anno_r'] = np.concatenate(
                ([ego_yaw], data['frame_anno_r']))
            data['frame_anno_t'] = np.concatenate(
                ([ego_trans], data['frame_anno_t']))
            data['frame_category_name'] = np.concatenate(
                ([0], data['frame_category_name']))

            # add agent to ego coord rotation matrix and translation
            agent2g_mat = self.rot2d(-data['frame_anno_r'])  # [A, 2, 2]
            global2ego_mat = make_2d_rotation_matrix(ego_yaw)  # [2, 2]
            global2ego_mat = global2ego_mat[None]
            agent2ego_mat = global2ego_mat @ agent2g_mat  # [A, 2, 2]
            data['frame_agent2ego_mat'] = agent2ego_mat
            agent_t = data['frame_anno_t']
            agent_t = agent_t[:, :2] - ego_trans[None, :2]  # [A, 2]
            agent_t = agent_t[:, None, :]
            agent_t = np.transpose(agent_t, (0, 2, 1))
            agent2ego_trans = global2ego_mat @ agent_t
            agent2ego_trans = np.transpose(agent2ego_trans, (0, 2, 1))
            data['frame_agent2ego_t'] = agent2ego_trans[:, 0]
            agent2ego_mat_inv = np.empty_like(agent2ego_mat)
            for i in range(agent2ego_mat_inv.shape[0]):
                agent2ego_mat_inv[i] = np.linalg.inv(agent2ego_mat[i])
            data['frame_ego2agent_mat'] = agent2ego_mat_inv
            agent_past = data['frame_past_traj']
            agent_past_mask = data['frame_past_traj_valid_mask']
            agent_fut = data['frame_fut_traj']
            agent_fut_mask = data['frame_fut_traj_valid_mask']
            # traj trans to agent_frame but no rot
            data['frame_past_traj_ego'] = self.trajs_transformer(
                agent_past, agent_past_mask, data['frame_anno_t'])
            data['frame_fut_traj_ego'] = self.trajs_transformer(
                agent_fut, agent_fut_mask, data['frame_anno_t'])

            # filter dataset: category, velocity, pc_range
            self.filter(data)

        if self.debug:
            print("[INFO]: Debug Mode ...")
            agent_past = data['past_trajs']
            agent_past = np.concatenate(
                (np.expand_dims(sdc_past_traj, axis=0), agent_past), axis=0)
            agent_past_mask = data['past_trajs_valid_mask']
            agent_past_mask = np.concatenate((np.expand_dims(
                sdc_past_traj_valid_mask, axis=0), agent_past_mask),
                                             axis=0)
            agent_fut = data['fut_trajs']
            agent_fut = np.concatenate(
                (np.expand_dims(sdc_fut_traj, axis=0), agent_fut), axis=0)
            agent_fut_mask = data['fut_trajs_valid_mask']
            agent_fut_mask = np.concatenate((np.expand_dims(
                sdc_fut_traj_valid_mask, axis=0), agent_fut_mask),
                                            axis=0)

            # 添加sdc到数据中
            # agent_past_angle = self.cal_veh_direc(agent_past)
            sample_token = data['frame_token']
            ego_trans = np.array(data['ego2global_translation'])
            ego_yaw = data['ego2global_yaw']

            agent_past = self.trajs_transformer(agent_past, agent_past_mask,
                                                ego_yaw, ego_trans)
            agent_fut = self.trajs_transformer(agent_fut, agent_fut_mask,
                                               ego_yaw, ego_trans)
            # transformer anchors
            # agent->global->ego
            data['frame_anno_r'] = np.concatenate(
                ([ego_yaw], data['frame_anno_r']))
            data['frame_anno_t'] = np.concatenate(
                ([ego_trans], data['frame_anno_t']))
            data['frame_category_name'] = np.concatenate(
                ([0], data['frame_category_name']))
            agent2g_mat = self.rot2d(-data['frame_anno_r'])  # [A, 2, 2]
            # agent2g_mat = agent2g_mat[:, None, None, :, :] # [A, 1, 1, 2, 2]
            # agent_t = data['frame_anno_t'][:, None, None, :2] # [A, 1, 1, 2]
            # transformerd_anchors = np.transpose(self.anchors.detach().cpu().numpy(),
            #                                (0, 1, 3, 2))[None].repeat(len(agent2g_mat), axis=0) # [A, 4, 6, 2, 12]
            # transformerd_anchors = np.matmul(agent2g_mat, transformerd_anchors)
            # transformerd_anchors = np.transpose(transformerd_anchors,
            #                                     (0, 1, 2, 4, 3))  # [A, 4, 6, 12, 2]
            # agent_categories = data['frame_category_name']
            # transformerd_anchors_grouped = []
            # agent_t_grouped = []
            # for i in range(len(agent_categories)):
            #     if agent_categories[i] == 3:
            #         continue
            #     transformerd_anchors_grouped.append(
            #         transformerd_anchors[i, agent_categories[i]])
            #     agent_t_grouped.append(agent_t[i])
            # transformerd_anchors = np.array(transformerd_anchors_grouped)  # [A, 6, 12, 2]
            # agent_t = np.array(agent_t_grouped) # [A, 1, 1, 2]
            # transformerd_anchors += agent_t
            # # global -> ego
            # transformerd_anchors -= ego_trans[None, None, None, :2]
            global2ego_mat = make_2d_rotation_matrix(ego_yaw)  # [2, 2]
            # global2ego_mat = global2ego_mat[None, None]
            # transformerd_anchors = np.transpose(transformerd_anchors, (0, 1, 3, 2))
            # transformerd_anchors = np.matmul(global2ego_mat, transformerd_anchors)
            # transformerd_anchors = np.transpose(transformerd_anchors, (0, 1, 3, 2))

            # method2: R_g-e @ R_a-e @ anchors + R_g-e@(T_a_g - T_e_g)
            global2ego_mat = global2ego_mat[None]
            agent2ego_mat = global2ego_mat @ agent2g_mat  # [A, 2, 2]
            data['agent2ego_mat'] = agent2ego_mat
            agent_t = data['frame_anno_t']
            agent_t = agent_t[:, :2] - ego_trans[None, :2]  # [A, 2]
            agent_t = agent_t[:, None, :]
            agent_t = np.transpose(agent_t, (0, 2, 1))
            agent2ego_trans = global2ego_mat @ agent_t
            agent2ego_trans = np.transpose(agent2ego_trans, (0, 2, 1))
            data['agent2ego_t'] = agent2ego_trans[:, 0]
            transformerd_anchors = np.transpose(
                self.anchors.detach().cpu().numpy(),
                (0, 1, 3, 2))[None].repeat(len(agent2ego_mat), axis=0)
            agent_categories = data['frame_category_name']
            transformerd_anchors_grouped = []
            agent2ego_mat_grouped = []
            agent2ego_trans_groouped = []
            for i in range(len(agent_categories)):
                if agent_categories[i] == 3:
                    continue
                transformerd_anchors_grouped.append(
                    transformerd_anchors[i, agent_categories[i]])
                agent2ego_mat_grouped.append(agent2ego_mat[i])
                agent2ego_trans_groouped.append(agent2ego_trans[i])
            agent2ego_trans = np.array(agent2ego_trans_groouped)
            agent2ego_mat = np.array(agent2ego_mat_grouped)
            transformerd_anchors = np.array(
                transformerd_anchors_grouped)  # [A, 6, 2, 12]
            # transformerd_anchors = np.transpose(transformerd_anchors, (0, 1, 3, 2))
            transformerd_anchors = np.matmul(agent2ego_mat[:, None, :, :],
                                             transformerd_anchors)
            transformerd_anchors = np.transpose(transformerd_anchors,
                                                (0, 1, 3, 2))
            transformerd_anchors += agent2ego_trans[:, None]

            # test re-trans
            transformerd_anchors -= agent2ego_trans[:, None]
            # test re-rot
            agent2ego_mat_inv = np.empty_like(agent2ego_mat)
            for i in range(agent2ego_mat_inv.shape[0]):
                agent2ego_mat_inv[i] = np.linalg.inv(agent2ego_mat[i])
            data['ego2agent_mat'] = agent2ego_mat_inv
            agent2ego_mat_inv = np.transpose(agent2ego_mat, (0, 2, 1))
            transformerd_anchors = np.transpose(transformerd_anchors,
                                                (0, 1, 3, 2))
            transformerd_anchors = np.matmul(agent2ego_mat_inv[:, None],
                                             transformerd_anchors)
            transformerd_anchors = np.transpose(transformerd_anchors,
                                                (0, 1, 3, 2))
            transformerd_anchors += agent2ego_trans[:, None]

            bev_render = BEVRender()
            bev_render.reset_canvas()
            bev_render.set_plot_cfg()
            bev_render.render_scene_trajs(
                sample_token,
                hist_traj=np.where(agent_past_mask, agent_past, np.nan),
                fut_traj=np.where(agent_fut_mask, agent_fut, np.nan),
                nusc=self.nusc,
                prediction_helper=self.pred_helper,
                nusc_maps=self.nusc_maps)
            # bev_render._render_traj(sdc_fut_traj, colormap='winter')
            # bev_render._render_traj(sdc_past_traj, colormap='gray')
            bev_render.render_transformed_anchors(transformerd_anchors)
            bev_render.save_fig("./result/debug.jpg")
            # viz anchor different anchor mode.
            # for i in range(len(self.anchors)):
            #     bev_render.reset_canvas()
            #     bev_render.set_plot_cfg()
            #     bev_render.render_anchors(self.anchors[i].detach().cpu().numpy())
            #     bev_render.save_fig(f'./result/debug_anchor_1_{i}.jpg')
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
            # sdc_fut_traj = convert_global_coords_to_local(
            #     coordinates=sdc_fut_traj,
            #     translation=ego_pose_start['translation'],
            #     rotation=ego_pose_start['rotation'],
            # )
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
            # sdc_past_traj = convert_global_coords_to_local(
            #     coordinates=sdc_past_traj,
            #     translation=ego_pose_start['translation'],
            #     rotation=ego_pose_start['rotation'],
            # )
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

    def cal_veh_direc(self, trajectory):
        # 计算x和y方向上的差分
        dx = np.diff(trajectory[:, :, 0], axis=1)
        dy = np.diff(trajectory[:, :, 1], axis=1)

        # 用arctan2得到方向的弧度表示，然后转换为度
        direction = np.rad2deg(np.arctan2(dy, dx))

        # 最后一个方向就是最后位置的速度方向，这是一个形状为(N,)的数组
        last_direction = direction[:, 0]
        return last_direction

    def load_annotation_file(self):
        if self.ann_file and 'pkl' in self.ann_file:
            with open(self.ann_file, 'rb') as f:
                print("[INFO] Loading pre-process file from {}".format(
                    self.ann_file))
                data = pickle.load(f)
                # sort data by timestamp
                data_infos = list(
                    sorted(data['infos'], key=lambda info: info['timestamp']))
                self.metadata = data['metadata']
                self.version = self.metadata['version']
                return data_infos
        else:
            raise ValueError("Please input .pkl pre-process file. \n \
                         u can generate by run './tools/nuscenes_create_data.sh'."
                             )

    @classmethod
    def trajs_transformer_to_ego(self,
                                 trajs,
                                 trajs_mask,
                                 yaw,
                                 trans,
                                 pred_mode=False):
        new_trajs = np.zeros_like(trajs)
        length = np.sum((np.sum(trajs_mask, axis=2) > 0), axis=1)  # [A,]
        for i in range(len(length)):
            if pred_mode:
                mode_traj = trajs[i]
                for j in range(len(mode_traj)):
                    new_trajs[i, j] = self.trans_global_local(
                        mode_traj[j], yaw, trans)
            else:
                new_trajs[i, :length[i]] = self.trans_global_local(
                    trajs[i, :length[i]], yaw, trans)
        return new_trajs

    def trajs_transformer(self, trajs, trajs_mask, trans):
        new_trajs = np.zeros_like(trajs)
        length = np.sum((np.sum(trajs_mask, axis=2) > 0), axis=1)  # [A,]
        for i in range(len(length)):
            new_trajs[i, :length[i]] = trajs[i, :length[i]] - trans[i][:2]
        return new_trajs

    @classmethod
    def trajs_transformer_to_global(self,
                                    trajs,
                                    trajs_mask,
                                    trans,
                                    pred_mode=False):
        new_trajs = np.zeros_like(trajs)
        length = np.sum((np.sum(trajs_mask, axis=2) > 0), axis=1)  # [A,]
        for i in range(len(length)):
            if pred_mode:
                new_trajs[i] = trajs[i] + trans[i][:2]
            else:
                new_trajs[i, :length[i]] = trajs[i, :length[i]] + trans[i][:2]
        return new_trajs

    @classmethod
    def trans_global_local(self, coordinates, yaw, translation):
        transform = make_2d_rotation_matrix(angle_in_radians=yaw)
        coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T
        res = np.dot(transform, coords).T[:, :2]
        return res

    def rot2d(self, yaw):
        siny, cosy = np.sin(yaw), np.cos(yaw)
        R = np.zeros((len(yaw), 2, 2))
        R[:, 0, 0] = cosy
        R[:, 0, 1] = -siny
        R[:, 1, 0] = siny
        R[:, 1, 1] = cosy
        return R

    def filter(self, data):
        """This func used to filter pc_range, velocity"""
        # filter range
        past_trajs = data['frame_past_traj_ego']
        x_in_range = np.logical_and(past_trajs[:, :, 0] >= self.pc_range[0],
                                    past_trajs[:, :, 0] <= self.pc_range[3])
        y_in_range = np.logical_and(past_trajs[:, :, 1] >= self.pc_range[1],
                                    past_trajs[:, :, 1] <= self.pc_range[4])
        in_rectangle = np.logical_and(x_in_range, y_in_range)
        track_in_rectangle = np.all(in_rectangle, axis=1)
        filter_index = track_in_rectangle

        # filter velocity
        velocity = copy.deepcopy(data['frame_velocity'])
        velocity.insert(0, np.array([np.inf, np.inf]))
        velocity_valid_index = np.zeros_like(filter_index)
        for i in range(len(velocity)):
            if np.sqrt(velocity[i][0]**2 + velocity[i][1]**2) > 0.3:
                velocity_valid_index[i] = 1
        frame_fut_traj_valid_mask = data['frame_fut_traj_valid_mask']
        frame_traj_velocity_valid_mask = np.ones_like(
            frame_fut_traj_valid_mask)
        for i in range(len(frame_fut_traj_valid_mask)):
            if velocity_valid_index[i] == False:
                frame_traj_velocity_valid_mask[i] = np.zeros((12, 2))
        data['velocity_valid_index'] = velocity_valid_index
        # wheather filter velocity before train
        if self.filter_velocity_before_train:
            filter_index = np.logical_and(filter_index, velocity_valid_index)
        else:
            data['frame_fut_traj_valid_mask'] = np.logical_and(
                frame_fut_traj_valid_mask,
                frame_traj_velocity_valid_mask).astype(np.int)

        filter_keys = [
            'frame_past_traj_ego',
            'frame_fut_traj_ego',
            'frame_ego2agent_mat',
            'frame_agent2ego_mat',
            'frame_agent2ego_t',
            'frame_fut_traj',
            'frame_fut_traj_valid_mask',
            'frame_category_name',
            'frame_anno_t',
            'frame_anno_r',
            'frame_past_traj',
            'frame_past_traj_valid_mask',
            # 'velocity_valid_index'
        ]
        for k in filter_keys:
            data[k] = data[k][filter_index]


if __name__ == "__main__":
    args = {}
    args[
        'root_dir'] = '/home/jerome.zhou/AV-EnvModeling-MotionForecasting/data/nuscenes/trainval/infos/nuscenes_v1.0-mini_infos_motionformer_train.pkl'
    args['load_from_disk'] = True
    dataset = MotionFormerDataset(args)
    for i in range(90, len(dataset)):
        dataset[i]
        pass
