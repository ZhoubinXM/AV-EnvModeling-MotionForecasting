from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from models.encoders.encoder import PredictionEncoder
from layer.sub_graph import SubGraph
from layer.global_graph import GlobalGraph
from train_eval.utils import *
from layer.base import MLP
from train_eval.utils import send_to_device, convert_double_to_float


class ImgBevEnc(PredictionEncoder):
    """Img and Bev feature encoder"""
    def __init__(self, args: dict):
        super(PredictionEncoder, self).__init__()
        self.use_img = args['use_img']
        self.use_subgraph = args['use_subgraph']
        backbone_name = args['backbone']
        train_backbone = args['train_backbone']
        return_interm_layer = args['return_interm_layer']
        dilation = args['dilation']
        bev_size = args['bev_feature_size']
        hidden_size = args['hidden_size']
        bev_layer_depth = args['depth']
        if self.use_img:
            self.backbone = Backbone(backbone_name, train_backbone,
                                    return_interm_layer, dilation)
            # Add a 1x1 convolution layer to reduce the channel dimension from 2048 to 256
            self.conv1x1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
            self.img_proj = nn.Linear(65536, 128)
        if self.use_subgraph:
            self.bev_enc = SubGraph(hidden_size)
            self.global_graph = GlobalGraph(hidden_size, num_attention_heads=4)
        else:
            self.bev_enc = BevEncoder(bev_size, hidden_size, bev_layer_depth)


    def forward(self, inputs: List[Dict], device: int = 0) -> Dict:
        # inputs = convert_double_to_float(send_to_device(inputs))
        if self.use_subgraph:
            pose_poly = inputs['obj_poly']
            pose_mask = inputs['obj_mask']
            pose_poly = pose_poly.reshape(pose_poly.shape[0], -1, pose_poly.shape[-1])
            pose_mask = pose_mask.reshape(pose_mask.shape[0], -1, pose_mask.shape[-1])
            bev_embedding = self.bev_enc(pose_poly, pose_mask, 0)
            bev_embedding = self.global_graph(pose_poly, pose_mask)
        else:
            past_position = inputs['target_history'].flatten(1).to(device)
            # past_position = inputs.flatten(1)
            bev_embedding = self.bev_enc(past_position)
        if self.use_img:
            img = inputs['img'].to(device)
            img_embedding = self.backbone(img)['0']
            # Apply the 1x1 convolution to img_embedding
            img_embedding = self.conv1x1(img_embedding)
            # Flatten the last three dimensions of img_embedding
            img_embedding = img_embedding.view(img_embedding.size(0), -1)
            img_embedding = self.img_proj(img_embedding)
        else:
            img_embedding = bev_embedding
        # Concatenate img_embedding and bev_embedding along the last dimension

        out = torch.cat((img_embedding, bev_embedding), dim=-1)
        return out


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3"
            }
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        out = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, train_backbone: bool,
                 return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels,
                         return_interm_layers)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BevEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size=128, depth=3) -> None:
        super().__init__()
        self.emb = MLP(hidden_size=feature_size, out_features=hidden_size)
        self.layers = nn.ModuleList(
            [MLP(hidden_size, hidden_size) for _ in range(depth - 1)])

    def forward(self, x):
        x = self.emb(x)
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x
