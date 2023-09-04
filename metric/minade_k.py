from metric.metric import Metric
from typing import Dict, Union
import torch
import numpy as np
from metric.utils import min_ade
from train_eval.utils import get_from_mapping


class MinADEK(Metric):
    """
    Minimum average displacement error for the top K trajectories.
    """
    def __init__(self, args: Dict):
        self.k = args['k']
        self.name = 'min_ade_' + str(self.k)

    def __call__(self, predictions: Dict, mapping: Union[Dict, torch.Tensor], device) -> torch.Tensor:
        """
        Compute MinADEK
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        traj: torch.Tensor = predictions[0]
        probs: torch.Tensor = predictions[1]
        if not isinstance(mapping, dict):
            gt_traj: list = get_from_mapping(mapping, 'labels')
            gt_traj = torch.tensor(np.array(gt_traj),
                                device=device,
                                dtype=torch.float)
        else:
            # gt_traj = mapping['target_future']
            gt_traj = mapping['frame_fut_traj_ego'].reshape(
                -1, traj.shape[-2], traj.shape[-1])
        # Useful params
        batch_size = probs.shape[0]
        num_pred_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        masks = gt_traj['masks'] if type(gt_traj) == dict and 'masks' in gt_traj.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)
        origin_masks = mapping['frame_fut_traj_valid_mask'].reshape(-1, traj.shape[-2], traj.shape[-1])
        masks = 1 - (torch.sum(origin_masks, dim=-1) > 0).int()  # 0 means valid [bs*A, sequence_length]

        min_k = min(self.k, num_pred_modes)

        _, inds_topk = torch.topk(probs, min_k, dim=1)
        batch_inds = torch.arange(batch_size).unsqueeze(1).repeat(1, min_k)
        traj_topk = traj[batch_inds, inds_topk]

        errs, _ = min_ade(traj_topk, gt_traj, masks)
        
        loss_sum_num = (torch.sum(1 - masks, dim=1).long() > 0).int().sum()
        # return torch.mean(errs)
        return torch.sum(errs) / loss_sum_num
    