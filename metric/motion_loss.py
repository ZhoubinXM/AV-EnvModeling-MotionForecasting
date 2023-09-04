import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from metric.metric import Metric
from train_eval.utils import get_from_mapping, get_dis_point_2_points

from typing import Dict

eps = 1e-5


class MotionLoss(Metric):
    """Variety loss"""
    def __init__(self):
        self.name = 'motion_loss'

    def __call__(self, predictions, mapping, device) -> Tensor:
        pred_traj: torch.Tensor = predictions[0]
        pred_prob = predictions[1]
        _, _, T, _ = pred_traj.shape
        if isinstance(mapping, dict):
            gt_traj = mapping['frame_fut_traj_ego'].reshape(
                -1, pred_traj.shape[-2], pred_traj.shape[-1])
        else:
            gt_traj = get_from_mapping(mapping, 'labels')
        if not isinstance(gt_traj, torch.Tensor):
            gt_traj_tensor = torch.tensor(np.array(gt_traj),
                                          device=device,
                                          dtype=torch.float)
        else:
            gt_traj_tensor = gt_traj
        origin_masks = mapping['frame_fut_traj_valid_mask'].reshape(
            -1, pred_traj.shape[-2], pred_traj.shape[-1])
        masks = 1 - (torch.sum(origin_masks, dim=-1) >
                     0).int()  # 0 means valid [bs*A, sequence_length]
        _, argmin = get_dis_point_2_points(gt_traj_tensor, pred_traj, masks)

        loss_ = F.smooth_l1_loss(pred_traj.gather(
            1,
            argmin.repeat(T, 2, 1, 1).permute(3, 2, 0, 1)).squeeze(1) *
                                 origin_masks,
                                 gt_traj_tensor * origin_masks,
                                 reduction='none')
        loss_sum_num = torch.sum((torch.sum(origin_masks, dim=-1) > 0).int())
        argmin = argmin[torch.sum(1 - masks, dim=1) > 0]
        pred_prob = pred_prob[torch.sum(1 - masks, dim=1) > 0]
        # loss_sum_num = pred_traj.shape[0] * pred_traj.shape[2]
        return loss_.sum() / (loss_sum_num) + F.nll_loss(pred_prob, argmin)
