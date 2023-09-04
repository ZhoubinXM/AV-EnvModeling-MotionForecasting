import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_adms_model, initialize_metric, \
    initialize_adms_dataset
import torch
import os
import train_eval.utils as u
import numpy as np
import json
from datasets.adms_dataset import adms_collate
from datasets.motionformer_dataset import MotionFormerDataset
from visualize.bev_render import BEVRender
from visualize.camera_render import CameraRender
from visualize.utils import combine
# for visualization nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global, convert_global_coords_to_local
from nuscenes.map_expansion.map_api import NuScenesMap

# Initialize device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """
    Class for evaluating trained models
    """
    def __init__(self, cfg: Dict, data_root: str, data_dir: str,
                 checkpoint_path: str):
        """
        Initialize evaluator object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        self.past_length = int(cfg['past_length'])
        self.pred_length = int(cfg['pred_length'])
        self.viz = cfg['viz']
        self.debug = cfg['debug']
        self.dataset = cfg['dataset']

        # TODO: Initialize test dataset
        test_set = initialize_adms_dataset(cfg['dataset'],
                                           cfg['val_datafile'],
                                           mode='val')

        # Initialize dataloader
        self.dl = torch_data.DataLoader(
            test_set,
            cfg['batch_size'],
            shuffle=False,
            # num_workers=cfg['num_workers'],
        )

        # Initialize model
        self.model = initialize_adms_model(
            cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
            cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(device)
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # # 使用torch.jit.load加载模型
        # loaded_model = torch.jit.load("mlp_resmlp_2_3.pt")

        # # 确保模型处于评估模式
        # loaded_model.eval()

        # # 创建输入张量
        # example_input = torch.rand(1, 5, 2)

        # # 使用加载的模型进行推断
        # with torch.no_grad():
        #     output = loaded_model(example_input)
        #     output_1 = self.model(example_input)

        # # 输出预测结果
        # print(output)
        # print(output_1)

        # Initialize metrics
        self.metrics = [
            initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
            for i in range(len(cfg['val_metrics']))
        ]

        # Viz
        if cfg['viz']:
            if self.debug:
                version = 'v1.0-mini'
            else:
                version = 'v1.0-trainval'
            self.k = 1
            dataroot = "./data/nuscenes/trainval"
            self.nusc = NuScenes(version=version,
                                 dataroot="./data/nuscenes/trainval",
                                 verbose=True)
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
            self.predict_helper = PredictHelper(self.nusc)

            self.bev_render = BEVRender()
            self.cam_render = CameraRender()

    def evaluate(self, output_dir: str):
        """
        Main function to evaluate trained model
        :param output_dir: Output directory to store results
        """
        self.output_dir = output_dir

        # Initialize aggregate metrics
        agg_metrics = self.initialize_aggregate_metrics()

        prediction_res = []
        label_res = []

        with torch.no_grad():
            for i, data in enumerate(self.dl):
                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))
                # data['target_history'] = data['target_history'][:, :self.past_length]
                # data['target_future'] = data['target_future'][:, :self.pred_length]
                # Forward pass
                predictions = self.model(data)

                # Aggregate metrics
                agg_metrics = self.aggregate_metrics(agg_metrics, predictions,
                                                     data)

                # Viz
                if self.viz:
                    if self.dataset == 'motion_former_dataset':
                        self.viz_motionformer(predictions, data, i)
                    else:
                        self.visualization(predictions, data, i)

                self.print_progress(i)

        # compute and print average metrics
        self.print_progress(len(self.dl))
        with open(os.path.join(output_dir, 'results', "results.txt"),
                  "w") as out_file:
            for metric in self.metrics:
                avg_metric = agg_metrics[str(
                    metric)] / agg_metrics['sample_count']
                output = str(metric) + ': ' + format(avg_metric, '0.2f')
                print(output)
                out_file.write(output + '\n')

    def initialize_aggregate_metrics(self):
        """
        Initialize aggregate metrics for test set.
        """
        agg_metrics = {'sample_count': 0}
        for metric in self.metrics:
            agg_metrics[str(metric)] = 0

        return agg_metrics

    def aggregate_metrics(self, agg_metrics: Dict, model_outputs: Dict,
                          ground_truth: Dict):
        """
        Aggregates metrics for evaluation
        """
        minibatch_metrics = {}
        for metric in self.metrics:
            minibatch_metrics[str(metric)] = metric(model_outputs,
                                                    ground_truth, device)

        masks = ground_truth['frame_fut_traj_valid_mask'].reshape(-1, 12, 2)
        masks = 1 - (torch.sum(masks, dim=-1) > 0).int()
        lengths = torch.sum(1 - masks, dim=1).long()
        batch_size = (lengths > 0).int().sum()
        agg_metrics['sample_count'] += batch_size

        for metric in self.metrics:
            agg_metrics[str(
                metric)] += minibatch_metrics[str(metric)] * batch_size

        return agg_metrics

    def print_progress(self, minibatch_count: int):
        """
        Prints progress bar
        """
        epoch_progress = minibatch_count / len(self.dl) * 100
        print('\rEvaluating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar,
              format(epoch_progress, '0.2f'),
              '%',
              end="\n" if epoch_progress == 100 else " ")

    def visualization(self, prediction, data, epoch):
        pred_trajs = prediction[0].cpu().detach().numpy()
        pred_probs = prediction[1].cpu().detach().numpy()
        bs = pred_trajs.shape[0]
        predict_num = pred_trajs.shape[1]
        for i in range(bs):
            self.bev_render.reset_canvas()
            self.bev_render.set_plot_cfg()
            hist_traj = data['target_history'][i].reshape(
                5, 2).cpu().detach().numpy()
            hist_traj = convert_local_coords_to_global(
                hist_traj,
                data['target_translation'][i].cpu().detach().numpy(),
                data['target_rotation'][i].cpu().detach().numpy())
            hist_traj = convert_global_coords_to_local(
                hist_traj, data['ego_translation'][i].cpu().detach().numpy(),
                data['ego_rotation'][i].cpu().detach().numpy())
            gt_traj = data['target_future'][i].reshape(
                6, 2).cpu().detach().numpy()
            gt_traj = convert_local_coords_to_global(
                gt_traj, data['target_translation'][i].cpu().detach().numpy(),
                data['target_rotation'][i].cpu().detach().numpy())
            gt_traj = convert_global_coords_to_local(
                gt_traj, data['ego_translation'][i].cpu().detach().numpy(),
                data['ego_rotation'][i].cpu().detach().numpy())
            for j in range(predict_num):
                pred_traj = pred_trajs[i][j]
                pred_traj = convert_local_coords_to_global(
                    pred_traj,
                    data['target_translation'][i].cpu().detach().numpy(),
                    data['target_rotation'][i].cpu().detach().numpy())
                pred_traj = convert_global_coords_to_local(
                    pred_traj,
                    data['ego_translation'][i].cpu().detach().numpy(),
                    data['ego_rotation'][i].cpu().detach().numpy())
                pred_trajs[i][j] = pred_traj

            self.bev_render.render_multi_pred_trajs(
                data['sample_token'][i],
                pred_trajs=pred_trajs[i],
                pred_scores=pred_probs[i],
                fut_traj=gt_traj,
                hist_traj=hist_traj,
                nusc=self.nusc,
                prediction_helper=self.predict_helper,
                nusc_maps=self.nusc_maps)

            file_path = os.path.join(
                self.output_dir, "figs",
                "pred_{}_{}_bev.jpg".format(data['instance_token'][i],
                                            data['sample_token'][i]))
            self.bev_render.save_fig(file_path)
            self.cam_render.reset_canvas()
            self.cam_render.render_image_data(data['sample_token'][i],
                                              self.nusc)
            file_path = os.path.join(
                self.output_dir, "figs",
                "pred_{}_{}.jpg".format(data['instance_token'][i],
                                        data['sample_token'][i]))
            self.bev_render.save_fig(file_path)
            combine(
                os.path.join(
                    self.output_dir, "figs",
                    "pred_{}_{}".format(data['instance_token'][i],
                                        data['sample_token'][i])))

    def viz_motionformer(self, prediction, data, epoch):
        pred_trajs = prediction[0].cpu().detach().numpy()  # [A, P, 12, 2]
        pred_probs = prediction[1] # [A, P]
        bs = pred_trajs.shape[0]
        predict_num = pred_trajs.shape[1]
        # viz param
        sample_token = data['frame_token'][0]  # 0 for bs=1
        agent_global_t = data['frame_anno_t'][0].cpu().detach().numpy()
        agent_yaw = data['frame_anno_r'][0].cpu().detach().numpy()
        ego_yaw = data['ego2global_yaw'].cpu().detach().numpy()[0]
        ego_trans = torch.tensor(
            data['ego2global_translation']).cpu().detach().numpy()

        # global coord used trans
        gt_hist_traj = data['frame_past_traj_ego'].reshape(
            -1, 7, 2).detach().cpu().numpy()
        gt_hist_traj_valid_mask = data['frame_past_traj_valid_mask'].reshape(
            -1, 7, 2).detach().cpu().numpy()
        gt_fut_traj = data['frame_fut_traj_ego'].reshape(
            -1, 12, 2).detach().cpu().numpy()
        gt_fut_traj_valid_mask = data['frame_fut_traj_valid_mask'].reshape(
            -1, 12, 2).detach().cpu().numpy()

        # convert to global coords.
        gt_hist_traj_global = MotionFormerDataset.trajs_transformer_to_global(
            gt_hist_traj, gt_fut_traj_valid_mask, agent_global_t)
        gt_fut_traj_gloabl = MotionFormerDataset.trajs_transformer_to_global(
            gt_fut_traj, gt_fut_traj_valid_mask, agent_global_t)
        pred_fut_traj_global = MotionFormerDataset.trajs_transformer_to_global(
            pred_trajs, gt_fut_traj_valid_mask, agent_global_t,
            pred_mode=True)  # [24, 6, 12, 2]
        # convert to ego coords.
        gt_hist_traj_ego = MotionFormerDataset.trajs_transformer_to_ego(
            gt_hist_traj_global, gt_hist_traj_valid_mask, ego_yaw, ego_trans)
        gt_fut_traj_ego = MotionFormerDataset.trajs_transformer_to_ego(
            gt_fut_traj_gloabl, gt_fut_traj_valid_mask, ego_yaw, ego_trans)
        pred_fut_traj_ego = MotionFormerDataset.trajs_transformer_to_ego(
            pred_fut_traj_global,
            gt_fut_traj_valid_mask,
            ego_yaw,
            ego_trans,
            pred_mode=True)
        
        # filter probs topk
        A, P, T, F = pred_fut_traj_ego.shape
        self.k = 6
        min_k = min(self.k, P)
        _, inds_topk = torch.topk(pred_probs, min_k, dim=1)
        inds_topk = inds_topk.cpu().detach().numpy()
        batch_inds = torch.arange(A).unsqueeze(1).repeat(1, min_k).cpu().detach().numpy()
        pred_fut_traj_ego = pred_fut_traj_ego[batch_inds, inds_topk]

        # render the sample
        self.bev_render = BEVRender()
        self.bev_render.reset_canvas()
        self.bev_render.set_plot_cfg()
        self.bev_render.render_scene_trajs(
            sample_token=sample_token,
            hist_traj=np.where(gt_hist_traj_valid_mask, gt_hist_traj_ego,
                               np.nan),
            fut_traj=np.where(gt_fut_traj_valid_mask, gt_fut_traj_ego, np.nan),
            nusc=self.nusc,
            prediction_helper=self.predict_helper,
            nusc_maps=self.nusc_maps)
        self.bev_render.render_transformed_anchors(pred_fut_traj_ego)
        file_path = os.path.join(self.output_dir, "figs",
                                 "{}.jpg".format(sample_token))
        self.bev_render.save_fig(file_path)
        self.bev_render.close_canvas()
