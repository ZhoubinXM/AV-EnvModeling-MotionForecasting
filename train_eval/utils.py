import math
import torch.optim
from typing import Dict, Union
import torch
import numpy as np


# Initialize device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def send_to_device(data: Union[Dict, torch.Tensor]):
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
