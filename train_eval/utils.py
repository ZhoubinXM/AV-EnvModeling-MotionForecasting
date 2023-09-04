import os
import sys
import math
import time
import pickle
import inspect
import logging
import logging.handlers
import torch.optim
from typing import Dict, Union, Optional, Tuple, Any, List
import torch
import numpy as np
import global_var as gv

import matplotlib
import matplotlib.pyplot as plt

import torch.distributed as dist

# Initialize device:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-5
anchor_info_path = './data/nuscenes/trainval/infos/motion_anchor_infos_mode6.pkl'

# Dist info
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def send_to_device(data: Union[Dict, torch.Tensor], device: Optional[int] = 0):
    """
    Send dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to(device)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v, device=device)
        return data
    else:
        return data

def convert_double_to_float(data: Union[Dict, torch.Tensor]):
    """
    Utility function to convert double tensors to float tensors in nested dictionary with Tensors
    """
    if type(data) is torch.Tensor and data.dtype == torch.float64:
        return data.float()
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert_double_to_float(v)
        return data
    else:
        return data

def convert_list_to_array(data):
    if type(data) is list:
        return np.array(data)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = convert_list_to_array(v)
        return data
    else:
        return data

def load_anchors():
    """
    Load the anchor information from a file.

    Args:
        anchor_info_path (str): The path to the file containing the anchor information.

    Returns:
        None
    """
    anchor_infos = pickle.load(open(anchor_info_path, 'rb'))
    return torch.stack(
        [torch.from_numpy(a) for a in anchor_infos['anchors_all']])  # Nc, Pc, steps, 2

def init_log(
        log_path: str,
        logger_name: Optional[str] = 'root',
        level: Optional[int] = logging.INFO,
        when: Optional[str] = 'D',
        backup: Optional[int] = 7,
        format:
    Optional[
        str] = "%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d:[%(module)s-%(funcName)s] -> %(message)s",
        datefmt: Optional[str] = "%m-%d %H:%M:%S"):
    """init log"""
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    handler = logging.handlers.TimedRotatingFileHandler(log_path,
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(log_path + '.wf',
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_color_text(text: str, color='red'):
    if color == 'red':
        return "\033[31m" + text + "\033[0m"
    else:
        assert False


def get_name(name=" ",
             mode: Optional[str] = "train",
             append_time: Optional[bool] = False):
    time_begin = gv.get_value('time_begin')
    if name.endswith(time_begin):
        return name
    prefix = mode + '_'
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


def get_time() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_angle(x: float, y: float) -> float:
    return math.atan2(y, x)


def rotate(x: float, y: float, angle: float) -> Tuple[float, float]:
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def larger(a: Any, b: Any, eps: Optional[float] = EPS):
    return a > b + eps


def get_pad_vector(li: list, hidden_size: Optional[int] = 128):
    """
    Pad vector to length of hidden_size
    """
    assert len(li) <= hidden_size
    li.extend([0] * (hidden_size - len(li)))
    return li


def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename,
              sys._getframe().f_back.f_lineno)
    assert satisfied


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(
        np.square((points[:, 0] - point_label[0])) +
        np.square((points[:, 1] - point_label[1])))


def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x**2 + der_y**2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)


def get_subdivide_points(polygon,
                         include_self=False,
                         threshold=1.0,
                         include_beside=False,
                         return_unit_vectors=False):
    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0])**2 +
                       (point_a[1] - point_b[1])**2)

    average_dis = 0
    for i, point in enumerate(polygon):
        if i > 0:
            average_dis += get_dis(point, point_pre)
        point_pre = point
    average_dis /= len(polygon) - 1

    points = []
    if return_unit_vectors:
        assert not include_self and not include_beside
        unit_vectors = []
    divide_num = 1
    while average_dis / divide_num > threshold:
        divide_num += 1
    for i, point in enumerate(polygon):
        if i > 0:
            for k in range(1, divide_num):

                def get_kth_point(point_a, point_b, ratio):
                    return (point_a[0] * (1 - ratio) + point_b[0] * ratio,
                            point_a[1] * (1 - ratio) + point_b[1] * ratio)

                points.append(get_kth_point(point_pre, point, k / divide_num))
                if return_unit_vectors:
                    unit_vectors.append(get_unit_vector(point_pre, point))
        if include_self or include_beside:
            points.append(point)
        point_pre = point
    if include_beside:
        points_ = []
        for i, point in enumerate(points):
            if i > 0:
                der_x = point[0] - point_pre[0]
                der_y = point[1] - point_pre[1]
                scale = 1 / math.sqrt(der_x**2 + der_y**2)
                der_x *= scale
                der_y *= scale
                der_x, der_y = rotate(der_x, der_y, math.pi / 2)
                for k in range(-2, 3):
                    if k != 0:
                        points_.append(
                            (point[0] + k * der_x, point[1] + k * der_y))
                        if i == 1:
                            points_.append((point_pre[0] + k * der_x,
                                            point_pre[1] + k * der_y))
            point_pre = point
        points.extend(points_)
    if return_unit_vectors:
        return points, unit_vectors
    return points
    # return points if not return_unit_vectors else points, unit_vectors


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        line_context = inspect.getframeinfo(
            inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]

def get_max_st_from_spans(spans: List[List[slice]]):
    spatial_num = []
    slice_num_list = []
    for _, batch_value in enumerate(spans):
        spatial_num.append(len(batch_value))
        slice_num = []
        for _, slice_value in enumerate(batch_value):
            slice_num.append(slice_value.stop - slice_value.start)
        slice_num_list.append(slice_num)
    return spatial_num, slice_num_list

def merge_tensors(tensors: List[torch.Tensor],
                  device,
                  hidden_size=None) -> Tuple[torch.Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    hidden_size = 128 if hidden_size is None else hidden_size
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def de_merge_tensors(tensor: torch.Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]


def get_dis_point_2_points(point, points):
    if points.ndim == 2:
        return np.sqrt(
            np.square(points[:, 0] - point[0]) +
            np.square(points[:, 1] - point[1]))
    elif points.ndim == 4:
        gt_traj = point.unsqueeze(1).repeat(1, 6, 1, 1)
        err = gt_traj - points[:, :, :, :2]
        # min ade
        # err = err[:, :, -1, :]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err, dim=2)
        err, idx = torch.min(err, dim=1)
        return err, idx
    
def get_dis_point_2_points(point, points, masks):
    """masks element 1 means invalid"""
    if points.ndim == 2:
        return np.sqrt(
            np.square(points[:, 0] - point[0]) +
            np.square(points[:, 1] - point[1]))
    elif points.ndim == 4:
        gt_traj = point.unsqueeze(1).repeat(1, 6, 1, 1)
        masks_rpt = masks.unsqueeze(1).repeat(1, 6, 1)
        err = gt_traj - points[:, :, :, :2]
        # min ade
        # err = err[:, :, -1, :]
        err = torch.pow(err, exponent=2)  # 计算每个元素的平方
        err = torch.sum(err, dim=-1)  # 计算平方后最后一维度的和
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err * (1 - masks_rpt), dim=2) / \
            torch.clip(torch.sum((1 - masks_rpt), dim=2), min=1)
        err, idx = torch.min(err, dim=1)
        return err, idx


def is_main_device(device, main_device: Optional[int] = 0) -> bool:
    return device == main_device


def show_heatmaps(matrices,
                  xlabel,
                  ylabel,
                  titles=None,
                  figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_attention-cues`"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=figsize,
                             sharex=True,
                             sharey=True,
                             squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    return fig


numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale  # 将位置向量缩放为弧度值，编码在sin，cos函数中的位置
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

def norm_points(pos, pc_range):
    """
    Normalize the end points of a given position tensor.

    Args:
        pos (torch.Tensor): Input position tensor.
        pc_range (List[float]): Point cloud range.

    Returns:
        torch.Tensor: Normalized end points tensor.
    """
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) 
    return torch.stack([x_norm, y_norm], dim=-1)

def anchor_coordinate_transform(anchors, mat, trans, with_trans=True, with_rot=True):
    G, P, T, F = anchors.shape
    bs, A, _, _ = mat.shape
    batch_anchors = []
    transformed_anchors = anchors[None, ...]  # [1, 4, 6, 12, 2]
    for i in range(bs):
        rot_mat = mat[i]
        t = trans[i]
        if with_rot:
            rot_mat = rot_mat[:, None, None, :, :]  # [A, 1, 1, 2, 2]
            from einops import rearrange, repeat
            # [1, 4, 6, 12, 2] -> [1, 4, 6, 2, 12]
            transformed_anchors = rearrange(transformed_anchors, 'b g m t c -> b g m c t')
            # [A, 4, 6, 2, 12]
            transformed_anchors = torch.matmul(rot_mat, transformed_anchors)
            # [A, 4, 6, 12, 2]
            transformed_anchors = rearrange(transformed_anchors, 'b g m c t -> b g m t c')
        if with_trans:
            transformed_anchors += t[:, None, None, None, :2]
        batch_anchors.append(transformed_anchors)
    return torch.stack(batch_anchors)

def traj_coordinate_transform(trajs, mat, trans, with_trans=True, with_rot=True):
    """Mat shape [B, A, 2, 2]; Trans shape [B, A, 2]"""
    B, A, P, T, F = trajs.shape  # [bs, A, 6, 12, 2]
    batch_trajs = []
    for i in range(B):
        rot_mat = mat[i]
        t = trans[i]
        transformed_trajs = trajs[i, ...]  # [A, 6, 12, 2]
        if with_rot:
            rot_mat = rot_mat[:, None, :, :]  # [A, 1, 2, 2]
            from einops import rearrange, repeat
            # [A, 6, 12, 2] -> [A, 6, 2, 12]
            transformed_trajs = rearrange(transformed_trajs, 'b m t c -> b m c t')
            # [A, 6, 2, 12]
            transformed_trajs = torch.matmul(rot_mat, transformed_trajs)
            # [A, 6, 12, 2]
            transformed_trajs = rearrange(transformed_trajs, 'b m c t -> b m t c')
        if with_trans:
            transformed_trajs += t[:, None, None, :2]
        batch_trajs.append(transformed_trajs)
    return torch.stack(batch_trajs)
