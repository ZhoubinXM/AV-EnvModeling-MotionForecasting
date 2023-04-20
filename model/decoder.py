import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetPred(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, m=50, device=torch.device("cpu")):
        """"""
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = m  # output candidate target
        self.device = device

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

    def target_nms(self, targets, threshold=1):
        target_selected = targets[:, :6]  # [batch_size, 1, horizon * 2]
        for batch_id in range(targets.shape[0]):  # one batch for a time
            cnt = 1
            for i in range(1, 50):
                dis = torch.pow(target_selected[batch_id, :cnt, :] - targets[batch_id, i], 2)
                dis = torch.sum(dis, dim=1)
                if not torch.any(dis < threshold):  # not exist similar trajectory
                    target_selected[batch_id, cnt] = targets[batch_id, i]  # add this trajectory
                    cnt += 1
                if cnt >= 6:
                    break
        return target_selected

    def forward(self, feat_in, tar_candidate, mask, candidate_gt=None, offset_gt=None):
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :return:
        """
        # tar_candidate = tar_candidate[:, :candidate_num, :]
        batch_size, n, _ = tar_candidate.size()

        # stack the target candidates to the end of input feature: [batch_size, n_tar, inchannels + 2]
        feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2)

        # compute probability for each candidate
        prob_tensor = self.prob_mlp(feat_in_repeat)
        prob_tensor += mask
        tar_candit_prob = F.softmax(prob_tensor, dim=1).squeeze(-1)  # [batch_size, n_tar]
        tar_offset = self.offset_mlp(feat_in_repeat)  # [batch_size, n_tar, 2]

        # TODO: 50 ...
        _, indices = torch.topk(tar_candit_prob, 50, dim=1)
        gather_indices = indices.unsqueeze(-1).repeat(1, 1, 2)
        tar_selected = torch.gather(tar_candidate, 1, gather_indices) + torch.gather(tar_offset, 1, gather_indices)
        tar_selected = self.target_nms(tar_selected, 2)

        loss = self.loss(tar_candit_prob, tar_offset, candidate_gt, offset_gt)
        return tar_selected, loss

    def loss(self, tar_candit_prob, tar_offset, candidate_gt, offset_gt, reduction="mean"):
        n_candidate_loss = F.binary_cross_entropy(tar_candit_prob, candidate_gt, reduction=reduction)
        # candidate_gt = candidate_gt.unsqueeze(-1).repeat(1, 1, 2)
        # tar_offset = torch.gather(tar_offset, 1, candidate_gt)
        offset_loss = F.smooth_l1_loss(tar_offset[candidate_gt.bool()], offset_gt, reduction=reduction)
        # print("target choose: ", n_candidate_loss.data, offset_loss.data, torch.max(tar_candit_prob, dim=1), tar_candit_prob[candidate_gt.bool()])
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

        # self.fc = nn.Linear(hidden_dim + 2, horizon * 2)
        # yapf: enable

    def forward(self, feat_in, loc_in, traj_gt, reduction="mean"):
        """
        predict the trajectory according to the target location
        :param feat_in: encoded feature vector for the target agent, torch.Tensor, [batch_size, in_channels]
        :param loc_in: end location, torch.Tensor, [batch_size, M, 2] or [batch_size, 1, 2]
        :return: [batch_size, M, horizon * 2] or [batch_size, 1, horizon * 2]
        """

        batch_size, M, _ = loc_in.size()
        if M > 1:
            # target candidates
            input = torch.cat([feat_in.repeat(1, M, 1), loc_in], dim=2)
        else:
            # targt ground truth
            input = torch.cat([feat_in, loc_in], dim=2)

        traj_pred = self.traj_pred(input)
        loss = F.smooth_l1_loss(traj_pred, traj_gt.repeat(1, M, 1), reduction=reduction)

        return traj_pred, loss


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

    # def forward(self, feat_in, traj_in, traj_gt, targets, reduction="mean"):
    #     """
    #     forward function
    #     :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
    #     :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
    #     :return: [batch_size, M]
    #     """
    #     batch_size, M, _ = traj_in.size()
    #     input_tenor = torch.cat([feat_in.repeat(1, M, 1), traj_in], dim=2)
    #     score_pred = F.softmax(self.score_mlp(input_tenor).squeeze(-1), dim=-1)
    #     score_gt = F.softmax(-self.distance_metric(traj_in, traj_gt) / self.temper, dim=1)
    #     score_gt = score_gt.detach()
    #     loss = F.binary_cross_entropy(score_pred, score_gt, reduction=reduction)
    #     selected_traj, targets_selected = self.traj_selection(traj_in, score_pred, targets, threshold=0.1)
    #     return selected_traj, targets_selected, loss

    def forward(self, feat_in, traj_in, traj_gt, targets, reduction="mean"):
        """
        forward function
        :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :return: [batch_size, M]
        """
        batch_size, M, _ = traj_in.size()
        input_tenor = torch.cat([feat_in.repeat(1, M, 1), traj_in], dim=2)
        score_pred = F.softmax(self.score_mlp(input_tenor).squeeze(-1), dim=-1)
        score_gt = F.softmax(-self.distance_metric(traj_in, traj_gt) / self.temper, dim=1)
        score_gt = score_gt.detach()
        loss = F.mse_loss(score_pred, score_gt)
        selected_traj, targets_selected = self.traj_selection(traj_in, score_pred, targets, threshold=0.1)
        return selected_traj, targets_selected, loss

class TNTDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, m=50, device=torch.device("cpu")):
        super(TNTDecoder, self).__init__()
        self.target_pred = TargetPred(in_channels, hidden_dim, m, device=device)
        self.motion_estimator = MotionEstimation(in_channels, horizon=30, hidden_dim=hidden_dim)
        self.traj_score_selection = TrajScoreSelection(in_channels, horizon=30, hidden_dim=hidden_dim)

    def forward(self, feat_in, tar_candidate, mask, traj_gt, candidate_gt, offset_gt):
        final_pos, loss_target_pred = self.target_pred(feat_in, tar_candidate, mask, candidate_gt, offset_gt)
        _, loss_motion_estimator = self.motion_estimator(feat_in, traj_gt[:, :, -2:], traj_gt)
        pred_traj, _ = self.motion_estimator(feat_in, final_pos, traj_gt)
        loss = 1 * loss_target_pred + loss_motion_estimator 
        return pred_traj, loss, final_pos

        # selected_traj, targets_selected, loss_traj_score_selection = self.traj_score_selection(
        #     feat_in, pred_traj, traj_gt, final_pos)
        # loss = 1 * loss_target_pred + loss_motion_estimator + 1 * loss_traj_score_selection
        # print(loss.data, 0.1 * loss_target_pred.data, loss_motion_estimator.data, 0.1 * loss_traj_score_selection.data)
        # return selected_traj, loss, targets_selected


if __name__ == "__main__":
    in_channels = 128
    horizon = 30
    device = torch.device("cuda")
    max_target = 2000
    model = TNTDecoder(in_channels=in_channels, hidden_dim=64, m=50, device=device)
    model.cuda()
    batch_size = 4

    feat_in = torch.randn((batch_size, 1, in_channels), device=device)
    tar_candidate = torch.rand((batch_size, max_target, 2), device=device)
    mask = torch.zeros((batch_size, max_target, 1), device=device)
    traj_gt = torch.randn((batch_size, 1, horizon * 2), device=device)
    candidate_gt = torch.zeros((batch_size, max_target), device=device)
    candidate_gt[:, 1] = 1
    offset_gt = torch.randn((batch_size, 2), device=device)

    # forward
    selected_traj, loss = model(feat_in, tar_candidate, mask, traj_gt, candidate_gt, offset_gt)
    print("selected_traj and loss: ", selected_traj, loss)
