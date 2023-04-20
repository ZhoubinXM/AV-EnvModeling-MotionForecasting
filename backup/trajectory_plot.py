import os, sys
from sklearn.feature_selection import f_oneway

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import train_eval.utils as u
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tkinter
import matplotlib.animation as animation
import time
from data_process.senario import *
# import matplotlib

# matplotlib.use('tkAgg')

senario = Senario()


def get_feature(row, idx):
    feature = {}
    feature['l_age'] = eval("row.l_age_" + str(idx))
    feature['l_pos_x'] = eval("row.l_pos_x_" + str(idx))
    feature['l_pos_y'] = eval("row.l_pos_y_" + str(idx))
    feature['l_vx'] = eval("row.l_vx_" + str(idx))
    feature['l_vy'] = eval("row.l_vy_" + str(idx))
    feature['l_ax'] = eval("row.l_ax_" + str(idx))
    feature['l_ay'] = eval("row.l_ay_" + str(idx))
    feature['c_obj_sts'] = eval("row.c_obj_sts_" + str(idx))
    feature['l_heading'] = eval("row.l_heading_" + str(idx))
    feature['l_length'] = eval("row.l_length_" + str(idx))
    feature['l_width'] = eval("row.l_width_" + str(idx))
    feature['c_main_class'] = eval("row.c_main_class_" + str(idx))
    feature['c_fusion_source'] = eval("row.c_fusion_source_" + str(idx))
    feature['l_dms_ptp'] = row.l_dms_ptp
    feature['l_VehSpdkph'] = row.l_VehSpdkph
    feature['l_Trvl'] = row.l_Trvl
    feature['l_StrWhlAgSAE'] = row.l_StrWhlAgSAE
    feature['l_StrWhlAgSpdSAE'] = row.l_StrWhlAgSpdSAE
    feature['l_TorsBarTqSAE'] = row.l_TorsBarTqSAE
    feature['l_LgtSAEAg'] = row.l_LgtSAEAg
    feature['l_LgtSAEAmpss'] = row.l_LgtSAEAmpss
    feature['l_LatSAEAg'] = row.l_LatSAEAg
    feature['l_LatSAEAmpss'] = row.l_LatSAEAmpss
    feature['l_YawRateSAERps'] = row.l_YawRateSAERps
    feature['c_dlb_uuid'] = row.c_dlb_uuid
    return feature


def get_inputs(file_name):
    df = pd.read_csv(file_name, low_memory=False)
    if "l_age_128" not in df.columns:
        print("file invalid!")
        return

    len = 0
    objs_of_sample = {}

    for row in df.itertuples():
        if row.c_dlb_uuid != '21b935a0-93d5-4a66-ab1e-10b986307d71':
            continue
        len += 1
        for idx in range(1, 129):
            obj_id = eval("row.c_obf_id_" + str(idx))
            if pd.notna(obj_id):
                if obj_id not in objs_of_sample.keys():
                    objs_of_sample[obj_id] = []

    for id in objs_of_sample.keys():
        objs_of_sample[id] = np.ones((len, 10)) * -1.0
    ego = np.ones((len, 6)) * -1.0

    time_idx = 0
    for row in df.itertuples():
        if row.c_dlb_uuid != '21b935a0-93d5-4a66-ab1e-10b986307d71':
            continue

        for idx in range(1, 129):
            obj_id = eval("row.c_obf_id_" + str(idx))
            feature = get_feature(row, idx)
            if pd.notna(obj_id) and feature['c_fusion_source'] in [4, 5, 6, 7]:
                obj_id = int(obj_id)
                if obj_id not in objs_of_sample.keys():
                    objs_of_sample[obj_id] = []
                frame_feature = [
                    feature['l_dms_ptp'], feature['l_pos_x'], -feature['l_pos_y'], -feature['l_heading'], feature['c_obj_sts'],
                    feature['c_main_class'], feature['l_length'], feature['l_width'], feature['l_age'], obj_id
                ]
                objs_of_sample[obj_id][time_idx] = frame_feature

        ego_feature = [feature['l_dms_ptp'], feature['l_VehSpdkph'], -feature['l_YawRateSAERps'], 0, 0, 0]
        ego[time_idx] = ego_feature

        time_idx += 1

    return objs_of_sample, ego


def all_plot(t, ax1, ax2, ax3, objs_of_sample, ego_of_sample):
    plot_senario(t, ax1, objs_of_sample, ego_of_sample)
    plot_traj(t, ax2, objs_of_sample, ego_of_sample)
    plot_best_snapshot(t, ax3, objs_of_sample, ego_of_sample)


def plot_best_snapshot(t, ax, objs_of_sample, ego_of_sample):
    ax.cla()
    #plot ego
    # ego_feature = ego_of_sample[t]
    # left_top_x = ego_feature[3] + 2.5
    # left_top_y = ego_feature[4] + 1
    # right_top_x = ego_feature[3] + 2.5
    # right_top_y = ego_feature[4] - 1
    # left_bottom_x = ego_feature[3] - 2.5
    # left_bottom_y = ego_feature[4] + 1
    # right_bottom_x = ego_feature[3] - 2.5
    # right_bottom_y = ego_feature[4] - 1
    # ax.plot([left_top_x, right_top_x, right_bottom_x, left_bottom_x, left_top_x],
    #          [left_top_y, right_top_y, right_bottom_y, left_bottom_y, left_top_y],
    #          '-',
    #          lw=2,
    #          c='r')
    # ax.plot(ego_of_sample[:t + 1, 3], ego_of_sample[:t + 1, 4])

    #plot objs
    for obj_id, obj in senario.best_snapshot.items():
        traj = obj.traj
        current_pos = traj[-1]
        ax.plot(current_pos.x, current_pos.y, 'o', lw=5, c='y')
        ax.text(current_pos.x, current_pos.y, str(obj_id))
        x = [point.x for point in traj]
        y = [point.y for point in traj]
        ax.plot(x, y, '-', lw=1, c='y')

    ax.set_xlim(ego_feature[3] - 150, ego_feature[3] + 175)
    ax.set_ylim(-20, 35)
    ax.set_title(str(senario.best_len))


def plot_senario(t, ax, objs_of_sample, ego_of_sample):
    ax.cla()
    current_points = {}
    for obj in objs_of_sample.values():
        if obj[t, 0] != -1 and obj[t, 5] in [1, 2]:
            current_points[obj[t, 9]] = Point(obj[t, 1], obj[t, 2],obj[t, 3], obj[t, 0])
            current_time = obj[t, 0]
    senario.update(current_time, current_points)
    print(senario.current_ids)

    #plot ego
    ego_feature = ego_of_sample[t]
    left_top_x = ego_feature[3] + 2.5
    left_top_y = ego_feature[4] + 1
    right_top_x = ego_feature[3] + 2.5
    right_top_y = ego_feature[4] - 1
    left_bottom_x = ego_feature[3] - 2.5
    left_bottom_y = ego_feature[4] + 1
    right_bottom_x = ego_feature[3] - 2.5
    right_bottom_y = ego_feature[4] - 1
    ax.plot([left_top_x, right_top_x, right_bottom_x, left_bottom_x, left_top_x],
            [left_top_y, right_top_y, right_bottom_y, left_bottom_y, left_top_y],
            '-',
            lw=2,
            c='r')
    ax.plot(ego_of_sample[:t + 1, 3], ego_of_sample[:t + 1, 4])

    #plot objs
    for obj_id, obj in senario.objs.items():
        traj = obj.traj
        current_pos = traj[-1]
        ax.plot(current_pos.x, current_pos.y, 'o', lw=5, c='y')
        ax.text(current_pos.x, current_pos.y, str(obj_id))
        x = [point.x for point in traj]
        y = [point.y for point in traj]
        ax.plot(x, y, '-', lw=1, c='y')

    time_array = time.localtime(current_time / 1e9)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
    time_str += str(round(current_time / 1e6) % 1000)
    ax.set_xlim(ego_feature[3] - 150, ego_feature[3] + 175)
    ax.set_ylim(-20, 35)


def plot_traj(t, ax, objs_of_sample, ego_of_sample):
    ax.cla()

    #plot ego
    ego_feature = ego_of_sample[t]
    left_top_x = ego_feature[3] + 2.5
    left_top_y = ego_feature[4] + 1
    right_top_x = ego_feature[3] + 2.5
    right_top_y = ego_feature[4] - 1
    left_bottom_x = ego_feature[3] - 2.5
    left_bottom_y = ego_feature[4] + 1
    right_bottom_x = ego_feature[3] - 2.5
    right_bottom_y = ego_feature[4] - 1
    ax.plot([left_top_x, right_top_x, right_bottom_x, left_bottom_x, left_top_x],
            [left_top_y, right_top_y, right_bottom_y, left_bottom_y, left_top_y],
            '-',
            lw=2,
            c='r')

    ax.plot(ego_of_sample[:t + 1, 3], ego_of_sample[:t + 1, 4])

    xdata = []
    ydata = []
    for obj in objs_of_sample.values():
        if obj[t, 0] != -1 and obj[t, 5] in [1, 2, 3, 4, 5]:
            xdata.append(obj[t, 1])
            ydata.append(obj[t, 2])
            ptp_ts = obj[t, 0]
            if obj[t, 5] in [1, 2]:
                # hsin = obj[t, 6] * 0.5 * math.sin(obj[t, 3])
                # hcos = obj[t, 6] * 0.5 * math.cos(obj[t, 3])
                # wsin = obj[t, 7] * 0.5 * math.sin(obj[t, 3])
                # wcos = obj[t, 7] * 0.5 * math.cos(obj[t, 3])
                # left_top_x = obj[t, 1] + hcos - wsin
                # left_top_y = obj[t, 2] + hsin + wcos
                # right_top_x = obj[t, 1] + hcos + wsin
                # right_top_y = obj[t, 2] + hsin - wcos
                # left_bottom_x = obj[t, 1] - hcos - wsin
                # left_bottom_y = obj[t, 2] - hsin + wcos
                # right_bottom_x = obj[t, 1] - hcos + wsin
                # right_bottom_y = obj[t, 2] - hsin - wcos
                # car_x = [left_top_x, right_top_x, right_bottom_x, left_bottom_x, left_top_x]
                # car_y = [left_top_y, right_top_y, right_bottom_y, left_bottom_y, left_top_y]
                # ax.plot(car_x, car_y, '-', lw=1, c='y')
                ax.plot(obj[t, 1], obj[t, 2], 'o', lw=5, c='y')
                ax.text(obj[t, 1], obj[t, 2], str(obj[t, 9]))

                # plot history traj
                traj_start_idx = 0
                for i in range(t, -1, -1):
                    if obj[i, 1] == -1:
                        traj_start_idx = i + 1
                        break
                ax.plot(obj[traj_start_idx:t + 1, 1], obj[traj_start_idx:t + 1, 2], '-', lw=1, c='y')

    time_array = time.localtime(ptp_ts / 1e9)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
    time_str += str(round(ptp_ts / 1e6) % 1000)
    ax.set_xlim(ego_feature[3] - 150, ego_feature[3] + 175)
    ax.set_ylim(-20, 35)


if __name__ == "__main__":
    objs_of_sample, ego_of_sample = get_inputs("/data/nio_dataset/accident/database/db_accident_20230112_prod_15.csv")

    current_heading = 0
    current_pos = [0, 0]
    prev_ptp = ego_of_sample[0, 0]
    for ego_feature in ego_of_sample[1:]:
        time_scope = (ego_feature[0] - prev_ptp) / 1e9
        prev_ptp = ego_feature[0]
        current_heading += time_scope * ego_feature[2]
        dis = ego_feature[1] * time_scope / 3.6
        # print(time_scope, dis)
        current_pos[0] += dis * math.cos(current_heading)
        current_pos[1] += dis * math.sin(current_heading)
        # print(time_scope, dis, current_pos[0], current_pos[1])
        ego_feature[3] = current_pos[0]
        ego_feature[4] = current_pos[1]
        ego_feature[5] = current_heading

    # print(ego_of_sample[0:10,3])
    # print(ego_of_sample[0:10,4])
    # print(objs_of_sample[65][0:10, 1:4])

    for obj in objs_of_sample.values():
        obj_traj = []
        idx_list = []
        for idx, feature in enumerate(obj):
            if feature[0] != -1 and feature[5] in [1, 2, 3, 4, 5]:
                heading = math.atan2(feature[2], feature[1]) + ego_of_sample[idx, 5]
                dis = math.sqrt(feature[1]**2 + feature[2]**2)
                feature[1] = ego_of_sample[idx, 3] + dis * math.cos(heading)
                feature[2] = ego_of_sample[idx, 4] + dis * math.sin(heading)
                feature[3] = feature[3] + ego_of_sample[idx, 5]
                idx_list.append(idx)
        if idx_list != []:
            obj_traj = obj[idx_list, 1:3]
    print(objs_of_sample[65][0:10, 1:4])


    x_min, y_min, x_max, y_max, frame = 0, 0, 0, 0, np.inf
    for obj in objs_of_sample.values():
        obj_min = obj.min(axis=0)
        obj_max = obj.max(axis=0)
        x_min = obj_min[1] if obj_min[1] < x_min else x_min
        x_max = obj_max[1] if obj_max[1] > x_max else x_max
        y_min = obj_min[2] if obj_min[2] < y_min else y_min
        y_max = obj_max[2] if obj_max[2] > y_max else y_max
        frame = obj.shape[0] if frame > obj.shape[0] else frame

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig3, ax3 = plt.subplots()
    ani = animation.FuncAnimation(fig,
                                  all_plot,
                                  frames=frame,
                                  fargs=(ax1, ax2, ax3, objs_of_sample, ego_of_sample),
                                #   blit=True,
                                  interval=33,
                                  repeat=False)
    # ani.save("curr.gif", fps=100)
    plt.show()

    #plot objs
    plt.cla()
    for obj_id, obj in senario.best_snapshot.items():
        traj = obj.traj
        current_pos = traj[-1]
        plt.plot(current_pos.x, current_pos.y, 'o', lw=5, c='y')
        plt.text(current_pos.x, current_pos.y, str(obj_id))
        x = [point.x for point in traj]
        y = [point.y for point in traj]
        plt.plot(x, y, '-', lw=1, c='y')

    plt.xlim(ego_feature[3] - 150, ego_feature[3] + 175)
    plt.ylim(-20, 35)
    plt.title(str(senario.best_len))
    plt.savefig("best_snapshot.jpg")
