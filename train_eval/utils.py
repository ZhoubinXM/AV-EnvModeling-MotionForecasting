import os
import math
import logging
import logging.handlers
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


def init_log(
        log_path,
        logger_name='root',
        level=logging.INFO,
        when='D',
        backup=7,
        format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d:[%(funcName)s] -> %(message)s",
        datefmt="%m-%d %H:%M:%S"):
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
