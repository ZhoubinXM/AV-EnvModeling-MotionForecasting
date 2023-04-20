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


class TNTDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, horizon=30, N=2000, M=50):
        super(TNTDecoder, self).__init__()
        self.N = N
        self.M = M
        self.target_pred = TargetPred(in_channels, hidden_dim, self.M)
        self.motion_estimator = MotionEstimation(in_channels, horizon=horizon, hidden_dim=hidden_dim)

    def forward(self, feat_in, tar_candidate, mask, gt_loss_param=None, is_train=True):
        if is_train:
            return self._train(feat_in, tar_candidate, mask, gt_loss_param)
        else:
            return self._inference(feat_in, tar_candidate, mask)

    def _train(self, feat_in, tar_candidate, mask, gt_loss_param):
        # stack the target candidates to the end of input feature: [batch_size, n_tar, inchannels + 2]
        feat_in=feat_in.unsqueeze(1)
        feat_in_repeat_n = feat_in.repeat(1, self.N, 1)
        feat_in_repeat_m = feat_in.repeat(1, self.M, 1)
        tar_selected, prob_selected, tar_prob, tar_offset = self.target_pred(feat_in_repeat_n,
                                                                             tar_candidate,
                                                                             mask,
                                                                             is_train=True)
        traj_train = self.motion_estimator(feat_in, gt_loss_param["traj_gt"][:, :, -2:])
        traj_pred = self.motion_estimator(feat_in_repeat_m, tar_selected)
        loss = self.loss(tar_prob, tar_offset, traj_train, gt_loss_param)
        return traj_pred, prob_selected, loss

    def _inference(self, feat_in, tar_candidate, mask):
        feat_in_repeat_n = feat_in.repeat(1, self.N, 1)
        feat_in_repeat_m = feat_in.repeat(1, self.M, 1)

        tar_selected, prob_selected = self.target_pred(feat_in_repeat_n, tar_candidate, mask, is_train=False)
        pred_traj = self.motion_estimator(feat_in_repeat_m, tar_selected)
        return pred_traj, prob_selected

    def loss(self, tar_prob, tar_offset, traj_train, gt_loss_param):
        target_pred_loss = self.target_pred.loss(tar_prob, tar_offset, gt_loss_param["candidate_gt"],
                                                 gt_loss_param["offset_gt"], gt_loss_param["reduction"])
        motion_estimator_loss = self.motion_estimator.loss(traj_train, gt_loss_param["traj_gt"], gt_loss_param["reduction"])
        return gt_loss_param["lamb1"] * target_pred_loss + gt_loss_param["lamb2"] * motion_estimator_loss
