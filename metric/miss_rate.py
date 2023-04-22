from metric.metric import Metric
from typing import Dict, Union, Tuple
import torch
import numpy as np
from metric.utils import miss_rate
from train_eval.utils import get_from_mapping


class MissRateK(Metric):
    """
    Miss rate for the top K trajectories.
    """

    def __init__(self, args: Dict):
        self.k = args['k']
        self.dist_thresh = args['dist_thresh']
        self.name = 'miss_rate_' + str(self.k)

    def __call__(self, predictions: Tuple[torch.Tensor], mapping: Union[Dict, torch.Tensor], device) -> torch.Tensor:
        """
        Compute miss rate
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        traj: torch.Tensor = predictions[0]
        probs: torch.Tensor = predictions[1]
        gt_traj: list = get_from_mapping(mapping, 'labels')
        gt_traj = torch.tensor(np.array(gt_traj),
                               device=device,
                               dtype=torch.float)

        # Useful params
        batch_size = probs.shape[0]
        num_pred_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        masks = gt_traj['masks'] if type(gt_traj) == dict and 'masks' in gt_traj.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)

        min_k = min(self.k, num_pred_modes)

        _, inds_topk = torch.topk(probs, min_k, dim=1)
        batch_inds = torch.arange(batch_size).unsqueeze(1).repeat(1, min_k)
        traj_topk = traj[batch_inds, inds_topk]

        return miss_rate(traj_topk, gt_traj, masks, dist_thresh=self.dist_thresh)