import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global, convert_global_coords_to_local
from visualize.base_render import BaseRender
from visualize.utils import obtain_map_info


class BEVRender(BaseRender):
    """
    Render class for BEV
    """
    def __init__(self,
                 figsize=(20, 20),
                 margin: float = 50,
                 view: np.ndarray = np.eye(4),
                 show_gt_boxes=False):
        super(BEVRender, self).__init__(figsize)
        self.margin = margin
        self.view = view
        self.show_gt_boxes = show_gt_boxes

    def set_plot_cfg(self):
        # self.axes.set_xlim([-self.margin, self.margin])
        # self.axes.set_ylim([-self.margin, self.margin])
        self.axes.set_aspect('equal')
        self.axes.grid(False)

    def render_sample_data(self, canvas, sample_token):
        pass

    def render_dataset_frame(self, data_info, only_fw=False, ego_coord=True):
        """This func use to render dataset frame data

    Args:
        data_infos (_type_): _description_
    """
        if only_fw:
            fut_trajs, past_trajs = [], []
            fw_data_info = data_info['cams']['CAM_FRONT']
            for anno in fw_data_info:
                fut_trajs.append(anno['fut_traj'])
                past_trajs.append(anno['past_traj'])
            fut_trajs = np.array(fut_trajs)
            past_trajs = np.array(past_trajs)
        else:
            fut_trajs: np.array = data_info['fut_trajs']
            past_trajs: np.array = data_info['past_trajs']

        self.render_group_trajs(past_trajs,
                                colormap='gray',
                                ego=data_info,
                                ego_coord=ego_coord)
        self.render_group_trajs(fut_trajs, ego=data_info, ego_coord=ego_coord)

        # render ego
        trans = data_info['ego2global_translation']
        if ego_coord:
            trans = convert_global_coords_to_local(
                trans[:2], data_info['ego2global_translation'],
                data_info['ego2global_rotation'])[0]
        self.axes.scatter(trans[0], trans[1], marker='*', color='red', s=300)
        if ego_coord:
            self.axes.set_xlim([-self.margin, self.margin])
            self.axes.set_ylim([-self.margin, self.margin])

    def render_anno_data(self, sample_token, nusc: NuScenes,
                         predict_helper: PredictHelper):
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(
        ), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        # return boxes respect to sensor coord.
        data_path, boxes, _ = nusc.get_sample_data(
            lidar_record, selected_anntokens=sample_record['anns'])
        for box in boxes:
            instance_token = nusc.get('sample_annotation',
                                      box.token)['instance_token']
            future_xy_local = predict_helper.get_future_for_agent(
                instance_token, sample_token, seconds=6, in_agent_frame=True)
            trans = box.center
            rot = Quaternion(matrix=box.rotation_matrix)
            if future_xy_local.shape[0] > 0:
                future_xy = convert_local_coords_to_global(
                    future_xy_local, trans, rot)
                # future_xy = future_xy_local
                future_xy = np.concatenate([trans[None, :2], future_xy],
                                           axis=0)
                c = np.array([0, 0.8, 0])
                box.render(self.axes, view=self.view, colors=(c, c, c))
                # self._render_traj(future_xy, line_color=c, dot_color=(0, 0, 0))
            past_xy_loocal = predict_helper.get_past_for_agent(
                instance_token, sample_token, seconds=3, in_agent_frame=True)
            if past_xy_loocal.shape[0] > 0:
                past_xy = convert_local_coords_to_global(
                    past_xy_loocal, trans, rot)
                # self._render_traj(past_xy, colormap='gray')
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])

    def render_hd_map(self, nusc, nusc_maps, sample_token):
        sample_record = nusc.get('sample', sample_token)
        sd_rec = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        info = {
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'scene_token': sample_record['scene_token']
        }

        layer_names = [
            'road_divider', 'road_segment', 'lane_divider', 'lane',
            'road_divider', 'traffic_light', 'ped_crossing'
        ]
        # layer_names = ['ped_crossing']
        map_mask = obtain_map_info(nusc,
                                   nusc_maps,
                                   info,
                                   patch_size=(102.4, 102.4),
                                   canvas_size=(1024, 1024),
                                   layer_names=layer_names)
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = map_mask[:, ::-1] > 0
        map_show = np.ones((1024, 1024, 3))
        map_show[map_mask[0], :] = np.array([1.00, 0.50, 0.31])
        map_show[map_mask[1], :] = np.array([159. / 255., 0.0, 1.0])
        self.axes.imshow(map_show,
                         alpha=0.2,
                         interpolation='nearest',
                         extent=(-51.2, 51.2, -51.2, 51.2))

    def _render_traj(self,
                     future_traj,
                     traj_score=1,
                     colormap=None,
                     points_per_step=20,
                     line_color=None,
                     dot_color=None,
                     dot_size=25):
        if colormap:
            total_steps = (len(future_traj) - 1) * points_per_step + 1
            dot_colors = matplotlib.colormaps[colormap](np.linspace(
                0, 1, total_steps))[:, :3]
            dot_colors = dot_colors*traj_score + \
                (1-traj_score)*np.ones_like(dot_colors)
            total_xy = np.zeros((total_steps, 2))
            for i in range(total_steps - 1):
                unit_vec = future_traj[i // points_per_step +
                                       1] - future_traj[i // points_per_step]
                total_xy[i] = (i/points_per_step - i//points_per_step) * \
                    unit_vec + future_traj[i//points_per_step]
            total_xy[-1] = future_traj[-1]
            self.axes.scatter(total_xy[:, 0],
                              total_xy[:, 1],
                              c=dot_colors,
                              s=dot_size)
        else:
            if isinstance(dot_color, np.ndarray):
                self.axes.scatter(future_traj[:, 0],
                                  future_traj[:, 1],
                                  marker='o',
                                  c=dot_color,
                                  s=dot_size)
                self.axes.plot(future_traj[:, 0],
                               future_traj[:, 1],
                               linestyle='-',
                               c=line_color)

    def render_sdc_car(self):
        sdc_car_png = cv2.imread('sources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))

    def render_legend(self):
        legend = cv2.imread('sources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        # target_image_shape = self.get_axes_size()
        # # 获取目标图像的宽度和高度
        # target_width, target_height = target_image_shape
        # # 获取图例的宽度和高度
        # legend_height, legend_width, _ = legend.shape
        # # 计算图例在目标图像中的位置
        # left = target_width - legend_width
        # right = target_width
        # bottom = target_height - legend_height
        # top = target_height
        self.axes.imshow(legend, extent=(23, 51.2, -50, -40))

    def render_group_trajs(self,
                           trajs,
                           ego,
                           colormap='winter',
                           ego_coord=True):
        for anno_traj in trajs:
            anno_traj = anno_traj[np.any(anno_traj != [0, 0], axis=-1)]
            if anno_traj.shape[0] > 0:
                if ego_coord:
                    anno_traj = convert_global_coords_to_local(
                        anno_traj, ego['ego2global_translation'],
                        ego['ego2global_rotation'])
                self._render_traj(anno_traj,
                                  colormap=colormap,
                                  dot_color=(0, 0, 0))

    def render_predict_trajs(self, sample_token: str, pred_traj: np.array,
                             hist_traj: np.array, fut_traj: np.array,
                             nusc: NuScenes, prediction_helper: PredictHelper,
                             nusc_maps):
        self.render_anno_data(sample_token, nusc, prediction_helper)
        # for pred_traj in pred_trajs:
        self._render_traj(pred_traj, colormap='autumn')
        self._render_traj(hist_traj, colormap='gray')
        self._render_traj(fut_traj, colormap='winter')
        self.render_sdc_car()
        self.render_hd_map(nusc, nusc_maps, sample_token)

    def render_multi_pred_trajs(self, sample_token: str, pred_trajs: np.array,
                                pred_scores, hist_traj: np.array,
                                fut_traj: np.array, nusc: NuScenes,
                                prediction_helper: PredictHelper, nusc_maps):
        self.render_anno_data(sample_token, nusc, prediction_helper)
        probabilities = pred_scores
        # 创建彩虹颜色映射
        # high prob purper, low prob red
        colormap = cm.get_cmap('rainbow')
        # 将概率值标准化到 [0, 1] 范围
        norm_probabilities = (probabilities - np.min(probabilities)) / (
            np.max(probabilities) - np.min(probabilities))
        colors = colormap(norm_probabilities)
        for i in range(len(pred_trajs)):
            self._render_traj(pred_trajs[i],
                              dot_color=colors[i],
                              line_color=colors[i])
        self._render_traj(hist_traj, colormap='gray')
        self._render_traj(fut_traj, colormap='winter')
        self.render_sdc_car()
        self.render_hd_map(nusc, nusc_maps, sample_token)
