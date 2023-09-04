import torch
from torch import nn
from models.encoders.img_bev import BevEncoder
from train_eval.utils import load_anchors, pos2posemb2d, norm_points, \
    anchor_coordinate_transform, traj_coordinate_transform
from nuscenes.prediction.helper import convert_local_coords_to_global, convert_global_coords_to_local,\
convert_global_coords_to_local_no_trans, convert_local_coords_to_global_no_trans


class MotionFormer(nn.Module):
    """MotionFormer"""
    def __init__(self, args: dict):
        super().__init__()
        bev_size = args['bev_feature_size']
        hidden_size = args['hidden_size']
        bev_layer_depth = args['depth']
        self.former_layer = 3
        self.embed_dims = hidden_size
        self.num_anchor = 6
        self.num_anchor_group = 4
        self.predict_step = 12
        self.num_reg_fcs = 2
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.debug = True

        self._init_layers()
        self.bev_enc = BevEncoder(bev_size, hidden_size, bev_layer_depth)
        self.boxes_query_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.learnable_motion_query_embedding = nn.Embedding(
            self.num_anchor * self.num_anchor_group, self.embed_dims)

        self.agent_level_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.intention_interaction_layers = IntentionInteraction()
        self.agent_interaction_layers = nn.ModuleList(
            [AgentInteraction() for i in range(self.former_layer)])

        self.scene_level_ego_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.scene_level_offset_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        # load anchor
        self.kmeans_anchors = load_anchors()

    def forward(self, data, device):
        # construct Q_a
        agent_past = data['frame_past_traj_ego']  # [bs, A, 7, 2] ego_centric
        B, A, T, F = agent_past.shape
        agent_past = agent_past.reshape(-1, T, F)  # [bs*(A), 7, 2]
        agent_past_flatten = agent_past.flatten(1)  # [bs*(A), 14]
        agent_query = self.bev_enc(agent_past_flatten)  # [bs*(A), 256] 包含时序信息
        agent_query = agent_query.reshape(B, A, -1) # [bs, A, 256]
        dtype = agent_query.dtype
        # anchor
        num_groups = self.kmeans_anchors.shape[0]

        # pos of the points of curr agents
        curr_ponts = agent_past.reshape(B, A, T, F)[:, :, 0, :]  # [bs, A, 2]
        curr_ponts_norm = norm_points(curr_ponts, self.pc_range)
        agent_query_pos = pos2posemb2d(curr_ponts_norm.to(device))  # [A*B, 256]
        agent_query_pos = self.boxes_query_embedding_layer(
            agent_query_pos).reshape(B, A, 256)  # [1, A, 256]

        # construct learneable query positional embedding
        # predict module
        learnable_query_pose = self.learnable_motion_query_embedding.weight.to(
            device)  # [(4*6), 256]
        learnable_query_pose = torch.stack(
            torch.split(learnable_query_pose, self.num_anchor,
                        dim=0))  # [4, 6, 256]

        # construct the agent level/scene-level query positional embedding
        # agent former
        agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(
            num_groups, self.num_anchor, self.predict_step, 2).detach()  # [4, 6, 12, 2]
        agent2ego_mat = data['frame_agent2ego_mat']  # [bs, A, 2, 2]
        agent2ego_t = data['frame_agent2ego_t']  # [bs, A, 2]
        # TODO: 验证旋转的正确性
        # 【bs, A, 4, 6, 12, 2】
        scene_level_ego_anchors = anchor_coordinate_transform(agent_level_anchors, agent2ego_mat, agent2ego_t)
        scene_level_offset_anchors = anchor_coordinate_transform(agent_level_anchors, agent2ego_mat, agent2ego_t, with_trans=False)
        
        # norm by pc range
        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)  # [4, 6, 12, 2]
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)  # [bs, A, 4, 6, 12, 2]
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)  # [bs, A, 4, 6, 12, 2]
        
        # embedding, we only use the last point of the anchor
        agent_level_embedding = self.agent_level_embedding_layer(
            pos2posemb2d(agent_level_norm[..., -1, :]))  # G, P, D: [4, 6, 256]
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
            pos2posemb2d(scene_level_ego_norm[..., -1, :]))  # B, A, G, P , D: [bs, A, 4, 6, 256]
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
            pos2posemb2d(scene_level_offset_norm[..., -1, :]))  # B, A, G, P , D [bs, A, 4, 6, 256]
        
        learnable_embed = learnable_query_pose[None, None, ...].expand(
            B, A, -1, -1, -1)  # [B, A, 4, 6, 256]
        agent_level_embedding = agent_level_embedding[None, None, ...].expand(B, A, -1, -1, -1) # [B,A,4,6,256]
        
        # grouped, only get current category
        agent_category = data['frame_category_name']  # [bs, A]
        scene_level_offset_anchors = self.group_mode_qury_pos(scene_level_offset_anchors,
                                                              agent_category)  # [bs, A, 6, 12, 2]

        # reference trajs
        reference_trajs = scene_level_offset_anchors.detach()

        agent_level_embedding = self.group_mode_qury_pos(agent_level_embedding,
                                                         agent_category)  # [bs, A, 6, 256]
        scene_level_ego_embedding = self.group_mode_qury_pos(scene_level_ego_embedding,
                                                             agent_category)  # [bs, A, 6, 256]
        scene_level_offset_embedding = self.group_mode_qury_pos(scene_level_offset_embedding,
                                                                agent_category)  # [bs, A, 6, 256]
        learnable_embed = self.group_mode_qury_pos(learnable_embed,
                                                   agent_category)  # [bs, A, 6, 256]

        B, _, P, D = agent_level_embedding.shape
        agent_query_bc = agent_query.unsqueeze(2).expand(-1, -1, P,
                                                         -1)  # (B, A, P, D) agent query 扩充模式维
        agent_query_pos_bc = agent_query_pos.unsqueeze(2).expand(
            -1, -1, P, -1)  # (B, A, P, D)

        # static intention embedding
        # imutable throughout all layers
        # 每个agent不同模式终点的位置编码
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)  # [bs, A, 6, 256]
        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed  # [bs, A, 6, 256]
        # this value used in deformable attention reference points, update by each layer
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed)  # [bs, A, 6, 256]
        intermediate = []
        intermediate_reference_trajs = []
        for layer_idx in range(self.former_layer):
            # fuser static and dynamic intention embedding
            # the dynamic intention embedding is the output of the previous layer, 
            # which is initialized with anchor embedding
            dynamic_query_embed = self.dynamic_embed_fuser(
                torch.cat([
                    agent_level_embedding, scene_level_offset_embedding,
                    scene_level_ego_embedding
                ],
                          dim=-1))  # [bs, A, 6, 256]
            # fuse static and dynamic intention embedding
            query_embed_intention = self.static_dynamic_fuser(
                torch.cat([static_intention_embed, dynamic_query_embed],
                          dim=-1))  # [bs, A, 6, 256]

            # fuse intention embed and query embed
            query_embed = self.in_query_fuser(
                torch.cat([query_embed, query_embed_intention], dim=-1)) # [bs, A, 6, 256]
            # interaction between agents.
            agent_query_embed = self.agent_interaction_layers[layer_idx](
                query_embed,
                agent_query,
                query_pos=agent_query_pos_bc,
                key_pos=agent_query_pos)  # [B, A, 6, 256]

            # interaction between agents and maps
            # TODO:
            # interaction with agents and bev features (goals)
            # implemented with deformable transformer
            # TODO:

            # fusing the embeddings from different interaction layers
            query_embed = [
                agent_query_embed, agent_query_bc + agent_query_pos_bc
            ]
            query_embed = torch.cat(query_embed, dim=-1) # [bs, A, 6, 512]
            query_embed = self.out_query_fuser(query_embed)  # [bs, A, 6, 256]

            # update reference trajectory
            tmp = self.traj_reg_branches[layer_idx](query_embed)
            bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
            tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)

            # we predict speed of trajectory and use cumsum trick to get the trajectory
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
            new_reference_trajs = torch.zeros_like(reference_trajs)
            new_reference_trajs = tmp[..., :2]
            reference_trajs = new_reference_trajs.detach()
            # update end points used in deformable attention.
            reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

            # update embedding, which is used in the next layer
            # only update the embedding of the last step, i.e. the goal
            ep_offset_embed = reference_trajs.detach()  # [bs, A, 6, 12, 2]
            ep_ego_embed = traj_coordinate_transform(
                reference_trajs,
                data['frame_ego2agent_mat'],
                data['frame_agent2ego_t'],
                with_trans=True,
                with_rot=False).detach()  # [bs A, 6, 12, 2]
            ep_agent_embed = traj_coordinate_transform(
                reference_trajs,
                data['frame_ego2agent_mat'],
                data['frame_agent2ego_t'],
                with_trans=False,
                with_rot=True).detach()

            agent_level_embedding = self.agent_level_embedding_layer(pos2posemb2d(
                norm_points(ep_agent_embed[..., -1, :], self.pc_range)))  # [bs A, 6, 256]
            scene_level_ego_embedding = self.scene_level_ego_embedding_layer(pos2posemb2d(
                norm_points(ep_ego_embed[..., -1, :], self.pc_range)))  # [bs A, 6, 256]
            scene_level_offset_embedding = self.scene_level_offset_embedding_layer(pos2posemb2d(
                norm_points(ep_offset_embed[..., -1, :], self.pc_range)))  # [bs A, 6, 256]

            intermediate.append(query_embed)
            intermediate_reference_trajs.append(reference_trajs)

        return torch.stack(intermediate)

    def group_mode_qury_pos(self, group_query, category):
        batch_size, num_agents = group_query.shape[:2]
        batched_mode_query_pos = []
        for i in range(batch_size):
            grouped_mode_query_pos = []
            for j in range(num_agents):
                grouped_mode_query_pos.append(group_query[i, j,
                                                          category[i][j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)

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
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.predict_step * 2))
        traj_reg_branch = nn.Sequential(*traj_reg_branch)

        def _get_clones(module, N):
            import copy
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.former_layer
        self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
        self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)


class IntentionInteraction(nn.Module):
    """
    Modeling the interaction between anchors
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first)

    def forward(self, query):
        B, A, P, D = query.shape
        # B, A, P, D -> B*A,P, D
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out


class AgentInteraction(nn.Module):
    """
    Modeling the interaction between the agents
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first)

    def forward(self, query, key, query_pos=None, key_pos=None):
        '''
        query: context query (B, A, P, D) 
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        '''
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        mem = key.expand(B * A, -1, -1)
        # N, A, P, D -> N*A, P, D
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query