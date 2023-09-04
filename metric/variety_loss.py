import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from metric.metric import Metric
from train_eval.utils import get_from_mapping, get_dis_point_2_points

from typing import Dict

eps = 1e-5


class VarietyLoss(Metric):
    """Variety loss"""
    def __init__(self):
        self.name = 'variety_loss'

    def __call__(self, predictions, mapping, device) -> Tensor:
        pred_traj = predictions[0]
        pred_prob = predictions[1]
        _, _, T, _ = pred_traj.shape
        if isinstance(mapping, dict):
            gt_traj = mapping['target_future']
        else:
            gt_traj = get_from_mapping(mapping, 'labels')
        # gt_traj_valid = get_from_mapping(mapping, 'labels_is_valid')
        # loss = torch.zeros(pred_traj.shape[0], device=device)
        # for i in range(pred_traj.shape[0]):
        #     assert gt_traj_valid[i][-1]
        #     gt_points = np.array(gt_traj[i]).reshape([30, 2])
        #     argmin = np.argmin(
        #         get_dis_point_2_points(
        #             gt_points[-1], np.array(pred_traj[i, :, -1, :].tolist())))

        #     loss_ = F.smooth_l1_loss(pred_traj[i, argmin],
        #                              torch.tensor(gt_points,
        #                                           device=device,
        #                                           dtype=torch.float),
        #                              reduction='none')
        #     loss_ = loss_ * torch.tensor(gt_traj_valid[i],
        #                                  device=device,
        #                                  dtype=torch.float).view(30, 1)
        #     if gt_traj_valid[i].sum() > eps:
        #         loss[i] += loss_.sum() / gt_traj_valid[i].sum()

        #     loss[i] += F.nll_loss(pred_prob[i].unsqueeze(0),
        #                           torch.tensor([argmin], device=device))
        if not isinstance(gt_traj, torch.Tensor):
            gt_traj_tensor = torch.tensor(np.array(gt_traj),
                                        device=device,
                                        dtype=torch.float)
        else:
            gt_traj_tensor = gt_traj
        _, argmin = get_dis_point_2_points(gt_traj_tensor, pred_traj)

        loss_ = F.smooth_l1_loss(pred_traj.gather(
            1,
            argmin.repeat(T, 2, 1, 1).permute(3, 2, 0, 1)).squeeze(1),
                                 gt_traj_tensor,
                                 reduction='none')

        return loss_.sum() / (pred_traj.shape[0] *
                              pred_traj.shape[2]) + F.nll_loss(
                                  pred_prob, argmin)
