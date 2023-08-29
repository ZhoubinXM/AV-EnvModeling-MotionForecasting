import torch
from torch import nn


class MotionMTP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dims = args['hidden_size']
        self.num_reg_fcs = 2
        self.former_layer = 3
        self.predict_steps = 6
        self._init_layers()
        self.unflatten_traj = nn.Unflatten(3, (self.predict_steps, 2))
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, inter_states):
        outputs_trajs = []
        outputs_traj_scores = []
        for lvl in range(inter_states.shape[0]):
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)

            # we use cumsum trick here to get the trajectory
            # tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)
            outputs_trajs.append(tmp)
        outputs_traj_scores = torch.stack(outputs_traj_scores)  # [num_layer, B, A+1, 6]
        outputs_trajs = torch.stack(outputs_trajs)  # [num_layer, B, A+1, 6, 6, 2]
        return outputs_trajs, outputs_traj_scores

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        traj_cls_branch = []
        traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs - 1):
            traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(nn.Linear(self.embed_dims, 1))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        traj_reg_branch = []
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_reg_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs - 1):
            traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_reg_branch.append(nn.ReLU())
        traj_reg_branch.append(nn.Linear(self.embed_dims, 6 * 2))
        traj_reg_branch = nn.Sequential(*traj_reg_branch)

        def _get_clones(module, N):
            import copy
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.former_layer
        self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
        self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)
