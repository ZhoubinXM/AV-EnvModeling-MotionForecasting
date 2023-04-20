from cProfile import label
import numpy as np
import pandas as pd
import os
from data_process.backup.config import *
from matplotlib import pyplot as plt


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
    feature['ttc'] = row.ttc
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
    return feature


def get_obj_map(df, skip=0, stride=1, max_len=20, skip2=0):
    df = df.reindex(index=df.index[::-1])
    skip_idx = 0
    stride_idx = 1
    last_dlb_id = -1
    last_ptp = -1
    all_map = []
    obj_len = 0
    obj_map = {}
    for row in df.itertuples():
        dlb_id = row.c_dlb_uuid
        ptp = row.l_dms_ptp
        if last_dlb_id == -1:
            last_dlb_id = dlb_id
            last_ptp = ptp
            continue
        if dlb_id == last_dlb_id and abs(last_ptp - ptp) <= 100000000:
            if skip_idx < skip or obj_len >= max_len or row.ttc == -1 or abs(last_ptp - ptp) <= 1000:
                skip_idx += 1
            else:
                if len(obj_map) == 0:
                    for idx in range(1, 129):
                        obj_id = eval("row.c_obf_id_" + str(idx))
                        feature = get_feature(row, idx)
                        if pd.notna(obj_id) and feature['c_obj_sts'] > 1 and feature['c_main_class'] in [1, 2, 3, 4, 5]:
                            obj_map[obj_id] = [feature]
                    obj_len += 1
                else:
                    if stride_idx == stride:
                        for idx in range(1, 129):
                            obj_id = eval("row.c_obf_id_" + str(idx))
                            feature = get_feature(row, idx)
                            if pd.notna(
                                    obj_id) and obj_id in obj_map and feature['c_obj_sts'] > 1 and feature['c_main_class'] in [
                                        1, 2, 3, 4, 5
                                    ]:
                                obj_map[obj_id].append(feature)
                        obj_len += 1
                        stride_idx = 0
                    stride_idx += 1
        else:
            if (len(obj_map) != 0):
                all_map.append(obj_map)
            obj_map = {}
            skip_idx = 0
            stride_idx = 0
            obj_len = 0
        last_dlb_id = dlb_id
        last_ptp = ptp
    return all_map


def get_data(all_map, time_range, feature_num):
    objs = []
    labels = []
    masks = []
    driver_dense_features = []
    for obj_map in all_map:
        obj_num = len(obj_map)
        obj_vector = np.zeros((obj_num, time_range - 1, feature_num))
        valid_obj = 0
        obj_vector_lens = []
        for i, obj in enumerate(obj_map.values()):
            label = obj[0]['ttc']
            polyline_len = len(obj) - 1
            for j in range(len(obj) - 1):
                if obj[j]['l_age'] < obj[j + 1]['l_age']:
                    polyline_len = j
                    break
                obj_vector[valid_obj][j][0] = obj[j]['l_pos_x']
                obj_vector[valid_obj][j][1] = obj[j]['l_pos_y']
                # obj_vector[valid_obj][j][2] = obj[j + 1]['l_pos_x']
                # obj_vector[valid_obj][j][3] = obj[j + 1]['l_pos_y']
                obj_vector[valid_obj][j][4] = obj[j]['l_vx']
                obj_vector[valid_obj][j][5] = obj[j]['l_vy']
                obj_vector[valid_obj][j][6] = obj[j]['l_ax']
                obj_vector[valid_obj][j][7] = obj[j]['l_ay']
                obj_vector[valid_obj][j][8] = obj[j]['l_heading']
                obj_vector[valid_obj][j][9] = obj[j]['c_main_class']
                obj_vector[valid_obj][j][10] = obj[j]['l_length']
                obj_vector[valid_obj][j][11] = obj[j]['l_width']

                # for debug
                if j == 0:
                    polyline_len = 1
                    break

            if polyline_len != 0:
                obj_vector_lens.append(polyline_len)
                valid_obj += 1
        if valid_obj == 0:
            # print("NO valid obj")
            continue

        if label >= 15:
            # print("ttc > 15")
            continue

        obj_vector = obj_vector[:valid_obj, :, :]
        mask = np.zeros([obj_vector.shape[0], obj_vector.shape[1], feature_num // 2])
        for i, polyline_len in enumerate(obj_vector_lens):
            mask_len = obj_vector.shape[1] - polyline_len
            if mask_len > 0:
                mask[i, polyline_len:, :] = np.ones([mask_len, feature_num // 2]) * (-10000.0)
        objs.append(obj_vector)
        masks.append(mask)
        labels.append(label)
        driver_dense_features.append([
            obj[0]['l_VehSpdkph'], obj[0]['l_Trvl'], obj[0]['l_StrWhlAgSpdSAE'], obj[0]['l_StrWhlAgSAE'], obj[0]['l_LgtSAEAg'],
            obj[0]['l_LgtSAEAmpss'], obj[0]['l_LatSAEAg'], obj[0]['l_LatSAEAmpss'], obj[0]['l_YawRateSAERps']
        ])
    return objs, masks, labels, driver_dense_features


def viz(filename, obj_vector, mask, label, driver_dense_feature):
    for i in range(len(obj_vector)):
        fig, ax = plt.subplots()
        for j in range(len(obj_vector[i])):
            poly_len = 0
            for k in range(19):
                if mask[i][j][k][0] == 0:
                    poly_len += 1
            plt.scatter(obj_vector[i][j][:poly_len][:, 0], obj_vector[i][j][:poly_len][:, 1], color="blue")
        plt.savefig("output/data_fig/" + filename + "_" + str(i) + ".png")
        plt.cla()
        # plt.plot(obj_vector[i][:,0,4])
        plt.plot(obj_vector[i][:, 0, 5])
        # plt.plot(obj_vector[i][:,0,6])
        # plt.plot(obj_vector[i][:,0,7])
        plt.savefig("output/data_fig/speed_" + filename + "_" + str(i) + ".png")


def main():
    data_root = "/data/nio_dataset/database_with_label"
    time_range = 20
    feature_num = 128
    obj_vectors = []
    masks = []
    labels = []
    driver_dense_features = []
    file_idx = 0
    for _, _, fs in os.walk(data_root):
        for f in fs:
            if f.endswith('.csv'):
                print("file_idx: ", file_idx, "file_name: ", f, end=" ")
                df = pd.read_csv(os.path.join(data_root, f), low_memory=False)
                if "l_age_128" not in df.columns:
                    continue
                # for skip in range(0, 5, 10):
                skip = 0
                all_map = get_obj_map(df=df, skip=skip, stride=5, max_len=time_range)
                obj_vector, mask, label, driver_dense_feature = get_data(all_map=all_map,
                                                                            time_range=time_range,
                                                                            feature_num=feature_num)

                if (len(obj_vector)) > 0:
                    obj_vectors += obj_vector
                    masks += mask
                    labels += label
                    driver_dense_features += driver_dense_feature

                # viz(f, obj_vector, mask, label, driver_dense_feature)
                    print(len(obj_vector), end=" ")
                print(len(obj_vectors))
                if file_idx % 50 == 0:
                    np.save("/data/nio_dataset/processed_data/obj_vector_0110.npy", np.array(obj_vectors, dtype=object))
                    np.save("/data/nio_dataset/processed_data/mask_0110.npy", np.array(masks, dtype=object))
                    np.save("/data/nio_dataset/processed_data/label_0110.npy", np.array(labels, dtype=object))
                    np.save("/data/nio_dataset/processed_data/driver_dense_feature_0110.npy",
                            np.array(driver_dense_features, dtype=object))
                # print("file_idx: ", file_idx, "file_name: ", f)
                # file_idx:  77 file_name:  db_elk_intervention_active_20220911_stg_20.csv
                file_idx += 1
                # if file_idx >= 100:
                #     break
    np.save("/data/nio_dataset/processed_data/obj_vector_0110.npy", np.array(obj_vectors, dtype=object))
    np.save("/data/nio_dataset/processed_data/mask_0110.npy", np.array(masks, dtype=object))
    np.save("/data/nio_dataset/processed_data/label_0110.npy", np.array(labels, dtype=object))
    np.save("/data/nio_dataset/processed_data/driver_dense_feature_0110.npy",
            np.array(driver_dense_features, dtype=object))
    print("finish")


def main1():
    data_root = "/data/nio_dataset/database_with_label"

    # f ="db_elk_intervention_active_20220911_stg_20.csv"
    # df = pd.read_csv(os.path.join(data_root, f))

    # print(df.columns)

    file_idx = 0
    for _, _, fs in os.walk(data_root):
        for f in fs:
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_root, f), low_memory=False)
                print(file_idx, " ", f, ": ", "l_age_128" in df.columns)
                file_idx += 1


if __name__ == "__main__":
    main()
