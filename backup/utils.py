import numpy as np
import math
def cal_angle(traj):
    span = traj[-6:]
    interval = 2
    angles = []
    for j in range(len(span)):
        if j + interval < len(span):
            der_x, der_y = span[j + interval, 0] - span[j, 0], span[j + interval, 1] - span[j][1]
            angles.append([der_x, der_y])
    angles = np.array(angles)
    der_x, der_y = np.mean(angles, axis=0)
    angle = -math.atan2(der_y, der_x) + math.radians(90)
    return angle


def cal_speed(traj):
    duration = len(traj) * 0.1 - 0.1
    distance = np.sqrt((traj[-1, 0] - traj[0, 0])**2 + (traj[-1, 1] - traj[0, 1])**2)
    speed = distance / duration
    return speed