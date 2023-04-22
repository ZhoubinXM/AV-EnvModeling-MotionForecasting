import os
import sys
import math
import time
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

# Initialize device:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-5


def send_to_device(data: Union[Dict, torch.Tensor], device: Optional[int] = 0):
    """
    Send dictionary with Tensors to GPU
    """
    if type(data) is torch.Tensor:
        return data.to(device)
    elif type(data) is dict:
        for k, v in data.items():
            data[k] = send_to_device(v)
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
        return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))
    elif points.ndim == 4:
        gt_traj = point.unsqueeze(1).repeat(1, 6, 1, 1)
        err = gt_traj - points[:, :, :, :2]
        err = err[:, :, -1, :]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        err, idx = torch.min(err, dim=1)
        return err, idx


def is_main_device(device):
    return isinstance(device, torch.device) or device == 0


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_attention-cues`"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
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
