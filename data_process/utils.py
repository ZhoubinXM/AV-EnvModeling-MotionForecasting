import numpy as np
import math

def cal_angle(traj):
    span = traj[-6:]
    interval = 2
    angles = []
    for j in range(len(span)):
        if j + interval < len(span):
            der_x, der_y = span[j + interval].x - span[j].x, span[j + interval].y - span[j].y
            angles.append([der_x, der_y])
    angles = np.array(angles)
    der_x, der_y = np.mean(angles, axis=0)
    angle = -math.atan2(der_y, der_x) + math.radians(90)
    return angle

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y