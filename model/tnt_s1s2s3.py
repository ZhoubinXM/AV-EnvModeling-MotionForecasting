import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetPred(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, M=50):
        """"""
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = M

        # yapf: disable
        self.prob_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.offset_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        # yapf: enable

    def forward(self, feat_in, tar_candidate, mask, is_train=True):
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :return:
        """
        feat_cat = torch.cat([feat_in, tar_candidate], dim=2)

        # compute probability for each candidate
        prob_tensor = self.prob_mlp(feat_cat)
        prob_tensor += mask
        tar_prob = F.softmax(prob_tensor, dim=1)  # [batch_size, n_tar]
        tar_offset = self.offset_mlp(feat_cat)  # [batch_size, n_tar, 2]

        prob_selected, indices = torch.topk(tar_prob, self.M, dim=1)
        gather_indices = indices.repeat(1, 1, 2)
        tar_selected = torch.gather(tar_candidate, 1, gather_indices) + torch.gather(tar_offset, 1, gather_indices)

        return (tar_selected, prob_selected, tar_prob, tar_offset) if is_train else (tar_selected, prob_selected)

    def loss(self, tar_prob, tar_offset, candidate_gt, offset_gt, reduction="mean"):
        n_candidate_loss = F.binary_cross_entropy(tar_prob.squeeze(2), candidate_gt, reduction=reduction)
        offset_loss = F.smooth_l1_loss(tar_offset[candidate_gt.bool()], offset_gt, reduction=reduction)
        return n_candidate_loss + offset_loss


class MotionEstimation(nn.Module):
    def __init__(self, in_channels, horizon=30, hidden_dim=64):
        """
        estimate the trajectories based on the predicted targets
        :param in_channels:
        :param horizon:
        :param hidden_dim:
        """
        super(MotionEstimation, self).__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # yapf: disable
        self.traj_pred = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, horizon * 2)
        )
        # yapf: enable

    def forward(self, feat_in, loc_in):
        input = torch.cat([feat_in, loc_in], dim=2)
        traj_pred = self.traj_pred(input)
        return traj_pred

    def loss(self, traj_pred, traj_gt, reduction="mean"):
        return F.smooth_l1_loss(traj_pred, traj_gt, reduction=reduction)


class TrajScoreSelection(nn.Module):
    def __init__(self, feat_channels, horizon=30, hidden_dim=64, temper=0.01):
        """
        init trajectories scoring and selection module
        :param feat_channels: int, number of channels
        :param horizon: int, prediction horizon, prediction time x pred_freq
        :param hidden_dim: int, hidden dimension
        :param temper: float, the temperature
        """
        super(TrajScoreSelection, self).__init__()
        self.feat_channels = feat_channels
        self.horizon = horizon
        self.temper = temper

        # yapf: disable
        self.score_mlp = nn.Sequential(
            nn.Linear(feat_channels + horizon * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        # yapf: enable

    def distance_metric(self, traj_candidate, traj_gt):
        """
        compute the distance between the candidate trajectories and gt trajectory
        :param traj_candidate: torch.Tensor, [batch_size, M, horizon * 2] or [M, horizon * 2]
        :param traj_gt: torch.Tensor, [batch_size, horizon * 2] or [1, horizon * 2]
        :return: distance, torch.Tensor, [batch_size, M] or [1, M]
        """
        _, M, horizon_2_times = traj_candidate.size()
        dis = torch.pow(traj_candidate - traj_gt, 2).view(-1, M, int(horizon_2_times / 2), 2)
        dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)
        return dis

    # todo: determine appropiate threshold
    def traj_selection(self, traj_in, score, targets, threshold=0.01):
        """
        select the top k trajectories according to the score and the distance
        :param traj_in: candidate trajectories, [batch, M, horizon * 2]
        :param score: score of the candidate trajectories, [batch, M]
        :param threshold: float, the threshold for exclude traj prediction
        :return: [batch_size, k, horizon * 2]
        """

        # for debug
        # return traj_in, targets

        def distance_metric(traj_candidate, traj_gt):
            if traj_candidate.dim() == 2:
                traj_candidate = traj_candidate.unsqueeze(1)
            _, M, horizon_2_times = traj_candidate.size()
            dis = torch.pow(traj_candidate - traj_gt, 2).view(-1, M, int(horizon_2_times / 2), 2)
            dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)
            return dis

        # re-arrange trajectories according the the descending order of the score
        M, k = 50, 6
        _, batch_order = score.sort(descending=True)
        traj_pred = torch.cat([traj_in[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, M, self.horizon * 2)
        targets_sorted = torch.cat([targets[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, M, 2)
        traj_selected = traj_pred[:, :k]  # [batch_size, 1, horizon * 2]
        # targets_selected = targets_sorted[:, :k]
        for batch_id in range(traj_pred.shape[0]):  # one batch for a time
            traj_cnt = 1
            for i in range(1, M):
                dis = distance_metric(traj_selected[batch_id, :traj_cnt, :], traj_pred[batch_id, i].unsqueeze(0))
                if not torch.any(dis < threshold):  # not exist similar trajectory
                    traj_selected[batch_id, traj_cnt] = traj_pred[batch_id, i]  # add this trajectory
                    # targets_selected[batch_id, traj_cnt] = targets_sorted[batch_id, i]
                    traj_cnt += 1

                if traj_cnt >= k:
                    break  # break if collect enough traj

            # no enough traj, pad zero traj
            if traj_cnt < k:
                traj_selected[:, traj_cnt:] = 0.0
                # targets_selected = targets_selected[:, :traj_cnt]

        # return traj_selected, targets_selected
        return traj_selected, targets_sorted

    def forward(self, feat_in, traj_in, traj_gt=None, is_train=True):
        if is_train:
            return self._train(feat_in, traj_in, traj_gt)
        else:
            return self._inference(feat_in, traj_in)

    def _train(self, feat_in, traj_in, traj_gt=None, is_train=True):
        """
        forward function
        :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :return: [batch_size, M]
        """
        # feat_in: (batch, M, 128)
        input_tenor = torch.cat([feat_in, traj_in], dim=2)
        score_pred = F.softmax(self.score_mlp(input_tenor).squeeze(-1), dim=-1)
        score_gt = F.softmax(-self.distance_metric(traj_in, traj_gt) / self.temper, dim=1)
        return score_pred, score_gt

    def _inference(self, feat_in, traj_in):
        input_tenor = torch.cat([feat_in, traj_in], dim=2)
        score_pred = F.softmax(self.score_mlp(input_tenor).squeeze(-1), dim=-1)
        return score_pred

    def loss(self, score_pred, score_gt):
        return F.kl_div(score_pred.log(), score_gt)


class TNTDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, horizon=30, N=2000, M=50):
        super(TNTDecoder, self).__init__()
        self.N = N
        self.M = M
        self.target_pred = TargetPred(in_channels, hidden_dim, self.M)
        self.motion_estimator = MotionEstimation(in_channels, horizon=horizon, hidden_dim=hidden_dim)
        self.traj_score_selection = TrajScoreSelection(in_channels, horizon=30, hidden_dim=hidden_dim, temper=1)

    def forward(self, feat_in, tar_candidate, mask, gt_loss_param=None, is_train=True):
        if is_train:
            return self._train(feat_in, tar_candidate, mask, gt_loss_param)
        else:
            return self._inference(feat_in, tar_candidate, mask)

    def _train(self, feat_in, tar_candidate, mask, gt_loss_param):
        # stack the target candidates to the end of input feature: [batch_size, n_tar, inchannels + 2]
        feat_in = feat_in.unsqueeze(1)
        feat_in_repeat_n = feat_in.repeat(1, self.N, 1)
        feat_in_repeat_m = feat_in.repeat(1, self.M, 1)
        tar_selected, prob_selected, tar_prob, tar_offset = self.target_pred(feat_in_repeat_n,
                                                                             tar_candidate,
                                                                             mask,
                                                                             is_train=True)
        traj_train = self.motion_estimator(feat_in, gt_loss_param["traj_gt"][:, :, -2:])
        traj_pred = self.motion_estimator(feat_in_repeat_m, tar_selected)
        score_pred, score_gt = self.traj_score_selection(feat_in_repeat_m, traj_pred, gt_loss_param["traj_gt"])

        loss = self.loss(tar_prob, tar_offset, traj_train, score_pred, score_gt, gt_loss_param)
        return traj_pred, score_pred, loss

    def _inference(self, feat_in, tar_candidate, mask):
        feat_in_repeat_n = feat_in.repeat(1, self.N, 1)
        feat_in_repeat_m = feat_in.repeat(1, self.M, 1)

        tar_selected, prob_selected = self.target_pred(feat_in_repeat_n, tar_candidate, mask, is_train=False)
        traj_pred = self.motion_estimator(feat_in_repeat_m, tar_selected)
        score_pred = self.traj_score_selection(feat_in_repeat_m, traj_pred, is_train=False)
        return traj_pred, score_pred

    def loss(self, tar_prob, tar_offset, traj_train, score_pred, score_gt, gt_loss_param):
        target_pred_loss = self.target_pred.loss(tar_prob, tar_offset, gt_loss_param["candidate_gt"],
                                                 gt_loss_param["offset_gt"], gt_loss_param["reduction"])
        motion_estimator_loss = self.motion_estimator.loss(traj_train, gt_loss_param["traj_gt"], gt_loss_param["reduction"])
        traj_score_selection_loss = self.traj_score_selection.loss(score_pred, score_gt)
        return (gt_loss_param["lamb1"] * target_pred_loss, gt_loss_param["lamb2"] * motion_estimator_loss,
                gt_loss_param["lamb3"] * traj_score_selection_loss)
