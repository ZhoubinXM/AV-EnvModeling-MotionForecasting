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
        gt_traj = get_from_mapping(mapping, 'labels')
        gt_traj_valid = get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(pred_traj.shape[0])

        for i in range(pred_traj.shape[0]):
            assert gt_traj_valid[i][-1]
            gt_points = np.array(gt_traj[i]).reshape([30, 2])
            argmin = np.argmin(
                get_dis_point_2_points(
                    gt_points[-1], np.array(pred_traj[i, :, -1, :].tolist())))

            loss_ = F.smooth_l1_loss(pred_traj[i, argmin],
                                     torch.tensor(gt_points,
                                                  device=device,
                                                  dtype=torch.float),
                                     reduction='none')
            loss_ = loss_ * torch.tensor(gt_traj_valid[i],
                                         device=device,
                                         dtype=torch.float).view(30, 1)
            if gt_traj_valid[i].sum() > eps:
                loss[i] += loss_.sum() / gt_traj_valid[i].sum()

            loss[i] += F.nll_loss(pred_prob[i].unsqueeze(0),
                                  torch.tensor([argmin], device=device))

        return loss.mean()
