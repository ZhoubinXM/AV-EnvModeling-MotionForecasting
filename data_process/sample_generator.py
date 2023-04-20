from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
import math
import copy
import numpy as np
from operator import attrgetter
import time
import matplotlib.pyplot as plt
from senario import * 



class SampleGenerator(object):
    def __init__(self, config) -> None:
        self.config = config

    def generate_sample(self, ego, objs, centerlines):
        """
            input:
                ego:
                objs:
                centerlines:
            output:
                polylines:
                polyline_lens: 
                attention_mask: 
                target_candidate: 
                target_candidate_mask:
                target_gt:
                traj_gt:
        """
        for obj in objs: 
            self.genegrate_obj_vector()


    def genegrate_obj_vector(self, obj: Obj, polylines: np.array, polyline_lens: np.array, id: int, cls: int) -> None:
        history_traj = obj.traj[-1:-60:-3]
        start_timestamp = (20 - len(history_traj)) * 0.1
        polyline_lens[id] = len(history_traj) - 1
        for j, point in enumerate(history_traj):
            x, y = point.x, point.y
            timestamp = j * 0.1 + start_timestamp
            if j > 0:
                vector = [point_pre[0], point_pre[1], x, y, timestamp, cls == 0, cls == 1, cls == 2, id, j]
                vector = self.get_pad_vector(vector, self.hidden_size)
                polylines[id, j - 1, :] = vector
            point_pre = [x, y]
    
    def get_pad_vector(self, li, hidden_size):
        """
        Pad vector to length of hidden_size
        """
        assert len(li) <= hidden_size
        li.extend([0] * (hidden_size - len(li)))
        return li