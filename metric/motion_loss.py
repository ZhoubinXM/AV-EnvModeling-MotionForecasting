import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from metric.metric import Metric
from train_eval.utils import get_from_mapping, get_dis_point_2_points

from typing import Dict, Tuple

eps = 1e-5


class MotionLoss(Metric):
    """Mootion loss"""
    def __init__(self):
        self.name = 'motion_loss'
        self.cls_loss_weight=0.5
        self.nll_loss_weight=0.5	
        self.loss_weight_minade=0.	
        self.loss_weight_minfde=0.25

    def __call__(self, predictions, mapping, device) -> Tensor:
        # get gt label
        all_agents_fut = mapping['agent_fut_feature']
        all_agents_fut_mask = mapping['agent_fut_feature_mask']
        sdc_fut = mapping['sdc_fut_traj_all'].unsqueeze(1)
        sdc_fut_mask = mapping['sdc_fut_traj_valid_mask_all'].unsqueeze(1)
        all_agents_fut = torch.cat([all_agents_fut, sdc_fut],
                                   dim=1)[..., :6, :]
        all_agents_fut_mask = torch.cat([all_agents_fut_mask, sdc_fut_mask],
                                        dim=1)[..., :6, :]
        all_agents_fut_mask = (torch.sum(all_agents_fut_mask, dim=-1) > 0).bool()

        pred_traj = predictions[0]
        pred_prob = predictions[1]
        LAYER, B, A, P, T, F = pred_traj.shape
        all_agents_fut = all_agents_fut.reshape(B * A, T, F)
        all_agents_fut_mask = 1 - all_agents_fut_mask.reshape(B * A, T).to(pred_traj.dtype)
        all_loss = []
        cls_loss = []
        reg_loss = []
        mr = []
        minade = []
        minfde = []
        for li in range(LAYER):
            pred_traj_i = pred_traj[li].reshape(B * A, P, T, F)
            pred_prob_i = pred_prob[li].reshape(B * A, P)
            l_minfde, inds = min_fde(pred_traj_i, all_agents_fut,
                                     all_agents_fut_mask)
            try:
                l_mr = miss_rate(pred_traj_i, all_agents_fut,
                                 all_agents_fut_mask)
            except:
                l_mr = torch.zeros_like(l_minfde)
            l_minade, inds = min_ade(pred_traj_i, all_agents_fut,
                                     all_agents_fut_mask)
            inds_rep = inds.repeat(T, F, 1, 1).permute(3, 2, 0, 1)
            traj_best = pred_traj_i.gather(1, inds_rep).squeeze(dim=1)
            l_reg = l_minade
            # Compute classification loss
            l_class = -torch.squeeze(pred_prob_i.gather(1, inds.unsqueeze(1)))
            l_reg = torch.sum(l_reg) / (B * A + 1e-5)
            l_class = torch.sum(l_class) / (B * A + 1e-5)
            l_minade = torch.sum(l_minade) / (B * A + 1e-5)
            l_minfde = torch.sum(l_minfde) / (B * A + 1e-5)
            loss = l_class * self.cls_loss_weight + l_reg * self.nll_loss_weight + \
                    l_minade * self.loss_weight_minade + \
                      l_minfde * self.loss_weight_minfde
            all_loss.append(loss)
            cls_loss.append(l_class)
            reg_loss.append(l_reg)
            minade.append(l_minade)
            minfde.append(l_minfde)
            mr.append(l_mr)
        return all_loss, cls_loss, reg_loss, minade, minfde

def min_fde(traj: torch.Tensor, traj_gt: torch.Tensor,
            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes final displacement error for the best trajectory is a set,
    with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape
    [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape
    [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error,
    shape [batch_size]
    """
    num_modes = traj.shape[1]
    lengths = torch.sum(1 - masks, dim=1).long()
    valid_mask = lengths > 0
    traj = traj[valid_mask]
    traj_gt = traj_gt[valid_mask]
    masks = masks[valid_mask]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    lengths = torch.sum(1 - masks, dim=1).long()
    inds = lengths.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(
        1, num_modes, 1, 2) - 1

    traj_last = torch.gather(traj[..., :2], dim=2, index=inds).squeeze(2)
    traj_gt_last = torch.gather(traj_gt_rpt, dim=2, index=inds).squeeze(2)

    err = traj_gt_last - traj_last[..., 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=2)
    err = torch.pow(err, exponent=0.5)
    err, inds = torch.min(err, dim=1)

    return err, inds


def miss_rate(traj: torch.Tensor,
              traj_gt: torch.Tensor,
              masks: torch.Tensor,
              dist_thresh: float = 2) -> torch.Tensor:
    """
    Computes miss rate for mini batch of trajectories,
    with respect to ground truth and given distance threshold
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory,
    shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth,
    shape [batch_size, sequence_length]
    :param dist_thresh: distance threshold for computing miss rate.
    :return errs, inds: errors and indices for modes with min error,
    shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(dist, exponent=2)
    dist = torch.sum(dist, dim=3)
    dist = torch.pow(dist, exponent=0.5)
    dist[masks_rpt.bool()] = -math.inf
    dist, _ = torch.max(dist, dim=2)
    dist, _ = torch.min(dist, dim=1)
    m_r = torch.sum(torch.as_tensor(dist > dist_thresh)) / len(dist)

    return m_r


def traj_nll(pred_dist: torch.Tensor, traj_gt: torch.Tensor,
             masks: torch.Tensor):
    """
    Computes negative log likelihood of ground truth trajectory under a
    predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the
    prediction horizon

    :param pred_dist: parameters of a bivariate Gaussian distribution,
    shape [batch_size, sequence_length, 5]
    :param traj_gt: ground truth trajectory,
    shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth,
    shape [batch_size, sequence_length]
    :return:
    """
    mu_x = pred_dist[:, :, 0]
    mu_y = pred_dist[:, :, 1]
    x = traj_gt[:, :, 0]
    y = traj_gt[:, :, 1]

    sig_x = pred_dist[:, :, 2]
    sig_y = pred_dist[:, :, 3]
    rho = pred_dist[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    nll = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) + torch.pow(sig_y, 2) *
         torch.pow(y - mu_y, 2) - 2 * rho * torch.pow(sig_x, 1) *
         torch.pow(sig_y, 1) * (x - mu_x) * (y - mu_y)) - \
        torch.log(sig_x * sig_y * ohr) + 1.8379

    nll[nll.isnan()] = 0
    nll[nll.isinf()] = 0

    nll = torch.sum(nll * (1 - masks), dim=1) / (torch.sum(
        (1 - masks), dim=1) + 1e-5)
    # Note: Normalizing with torch.sum((1 - masks), dim=1) makes values
    # somewhat comparable for trajectories of
    # different lengths

    return nll


def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor,
            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average displacement error for the best trajectory is a set,
    with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape
    [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape
    [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape
    [batch_size]
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / \
        torch.clip(torch.sum((1 - masks_rpt), dim=2), min=1)
    err, inds = torch.min(err, dim=1)

    return err, inds
