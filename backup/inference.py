import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import train_eval.utils as u
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


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
    # feature['ttc'] = row.ttc
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


def get_model(model_file):
    model = torch.load(model_file)
    model = model.to("cpu")
    model.eval()
    return model


def get_inputs(file_name):
    df = pd.read_csv(file_name, low_memory=False)
    if "l_age_128" not in df.columns:
        print("file invalid!")
        return

    inputs = []
    dlb_id = "default"
    for row in df.itertuples():
        # if row.c_dlb_uuid != 'f043762e-c52a-4a52-9117-e29bd51dbae8':
        #     continue
        polylines = np.zeros((256, 19, 128))
        attention_mask = np.ones((256, 19, 64)) * (-10000.0)
        polynum = 0

        ttc_cal = []
        for idx in range(1, 129):
            obj_id = eval("row.c_obf_id_" + str(idx))
            feature = get_feature(row, idx)
            if pd.notna(obj_id) and feature['c_obj_sts'] > 1 and feature['c_main_class'] in [1, 2, 3, 4, 5]:
                obj_vector = np.zeros((19, 128))
                obj_vector[0][0] = feature['l_pos_x']
                obj_vector[0][1] = feature['l_pos_y']
                obj_vector[0][4] = feature['l_vx']
                obj_vector[0][5] = feature['l_vy']
                obj_vector[0][6] = feature['l_ax']
                obj_vector[0][7] = feature['l_ay']
                obj_vector[0][8] = feature['l_heading']
                obj_vector[0][9] = feature['c_main_class']
                obj_vector[0][10] = feature['l_length']
                obj_vector[0][11] = feature['l_width']
                polylines[polynum, :, :] = obj_vector
                attention_mask[polynum, 0, :] = np.zeros((1, 64))
                polynum += 1

                # for ttc
                obf_info = {
                    'x': feature['l_pos_x'],
                    'y': feature['l_pos_y'],
                    'vx': feature['l_vx'],
                    'vy': feature['l_vy'],
                    'length': feature['l_length'],
                    'width': feature['l_width'],
                    'heading': feature['l_heading'],
                    'obj_main_class': feature['c_main_class']
                }
                ttc_cal.append(get_obj_ttc(obf_info, feature['l_VehSpdkph']))
        driver_dense_features = np.array([[
            feature['l_VehSpdkph'], feature['l_Trvl'], feature['l_StrWhlAgSpdSAE'], feature['l_StrWhlAgSAE'],
            feature['l_LgtSAEAg'], feature['l_LgtSAEAmpss'], feature['l_LatSAEAg'], feature['l_LatSAEAmpss'],
            feature['l_YawRateSAERps']
        ]])
        input = {
            'veh_cate_features': np.zeros((1, 4), dtype=np.int32),
            'veh_dense_features': np.zeros((1, 4)),
            'driver_cate_features': np.zeros((1, 4), dtype=np.int32),
            'driver_dense_features': driver_dense_features,
            'polylines': polylines,
            'polynum': np.array([polynum]),
            'attention_mask': attention_mask,
            # 'label': feature['ttc'],
            'ttc_cal': ttc_cal,
            'dlb_id': feature['c_dlb_uuid'],
            'ptp_ts': feature['l_dms_ptp'],
            'speed': feature['l_VehSpdkph']
        }

        # if dlb_id != "default" and dlb_id != feature['c_dlb_uuid']:
        #     break
        # else:
        #     dlb_id = feature['c_dlb_uuid']
        if polynum != 0:
            inputs.append(input)

    return inputs


def do_inference(model, input):

    (veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
     attention_mask) = model.preprocess_inputs(input)
    ttc = model(veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
                attention_mask)
    return ttc


def get_obj_ttc(obf_info: dict, spdkph: float) -> float:
    EGO_Rear2Frnt = 4.0
    EGO_Center2Side = 2.0

    ego_vx = spdkph / 3.6

    try:
        dx = obf_info['x']
        dy = -obf_info['y']
        vx = obf_info['vx']
        vy = -obf_info['vy']
    except Exception:
        print(obf_info)

    if obf_info['obj_main_class'] in [1, 2]:
        # with size
        length = obf_info['length']
        width = obf_info['width']
        heading = -obf_info['heading']
        long_r2f = dx - length * 0.5 * math.cos(heading) - EGO_Rear2Frnt
        long_f2r = dx + length * 0.5 * math.cos(heading)
        long_r2r = dx - length * 0.5 * math.cos(heading)
        long_f2f = dx + length * 0.5 * math.cos(heading) - EGO_Rear2Frnt
        lat_l2l = dy + width * 0.5 * math.sin(heading) - EGO_Center2Side
        lat_l2r = dy + width * 0.5 * math.sin(heading) + EGO_Center2Side
        lat_r2l = dy - width * 0.5 * math.sin(heading) - EGO_Center2Side
        lat_r2r = dy - width * 0.5 * math.sin(heading) + EGO_Center2Side
        lat_collision_thrds = 0.5
    elif obf_info['obj_main_class'] in [3, 4, 5]:
        # no size
        long_r2f = dx - EGO_Rear2Frnt
        long_f2f = long_r2f
        long_f2r = dx
        long_r2r = long_f2r
        lat_l2l = dy - EGO_Center2Side
        lat_r2l = lat_l2l
        lat_l2r = dy + EGO_Center2Side
        lat_r2r = lat_l2r
        lat_collision_thrds = 0.
    else:
        return np.inf

    # long-ttc
    long_dst = np.array([long_r2f, long_f2r, long_r2r, long_f2f])
    lat_dst = np.array([lat_l2l, lat_l2r, lat_r2l, lat_r2r])
    long_ttc = -long_dst / (vx - ego_vx)
    lat_ttc = -lat_dst / vy
    # filter invalid ttc
    long_ttc[long_ttc[:] < 0] = 0
    lat_ttc[lat_ttc[:] < 0] = 0
    # long_ttc = long_ttc[long_ttc[:] > 0]
    # lat_ttc = lat_ttc[lat_ttc[:] > 0]
    if math.fabs(dy) <= EGO_Center2Side and math.fabs(vy) < 0.5:
        # for in lane obj, only cal with long_ttc
        # OV
        lat_ttc = long_ttc
    if min(lat_dst) < lat_collision_thrds and max(lat_dst) > -lat_collision_thrds and math.fabs(vy) < 0.5:
        # lat overlap & small lat_v
        # OV
        lat_ttc = long_ttc
    if math.fabs(dy) <= EGO_Center2Side and long_f2r < 0:
        # obj in lane and behind ego -> no risk
        long_ttc = []
        lat_ttc = []
    if len(long_ttc) > 0 and len(lat_ttc) > 0:
        # valid long & lat ttc
        if long_ttc.max() == 0 or lat_ttc.max() == 0 or long_ttc.max() < lat_ttc.min() or long_ttc.min() > lat_ttc.max():
            return np.inf
        else:
            return max(lat_ttc.min(), long_ttc.min())
    else:
        return np.inf


if __name__ == "__main__":
    model_file = "output/model/best_adms_model.pth"
    model = get_model(model_file)
    inputs = get_inputs("/data/nio_dataset/database_with_label/db_elk_intervention_active_20221113_prod_27.csv")
    # inputs = get_inputs("/data/nio_dataset/accident/database/db_accident_20230112_prod_15.csv")
    accident = {}
    for input in inputs:
        for key, value in input.items():
            if key in [
                    'veh_cate_features', 'veh_dense_features', 'driver_cate_features', 'driver_dense_features', 'polylines',
                    'polynum', 'attention_mask'
            ]:
                input[key] = torch.from_numpy(value)
        input = u.convert_double_to_float(input)
        ttc = do_inference(model, input)

        pos = input['polylines'][:input['polynum'], 0, :2].detach().numpy().squeeze().reshape(-1, 2)
        dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        # ttc_logic = min(min(input['ttc_cal']), 15)
        ttc_logic = min(input['ttc_cal'])
        current_data = np.array([ttc.detach().numpy().squeeze(), ttc_logic, min(dist), input['speed'], input['ptp_ts']])
        if input['dlb_id'] not in accident:
            accident[input['dlb_id']] = current_data
        else:
            accident[input['dlb_id']] = np.vstack((accident[input['dlb_id']], current_data))
        # print("-------------------")
        # print(ttc.detach().numpy().squeeze(), '\n', '\n', input['ttc_cal'], '\n')
    # print(accident)

    # plot
    crash_time = 6
    for dlb_id, data in accident.items():
        data = np.array(data)
        time = (data[:, 4] - data[0, 4]) / 1e9
        fig, axs = plt.subplots(3, 1)
        axs[0].set_title("dlb uuid: " + dlb_id)
        axs[0].plot(time, data[:, 0])
        # axs[0].axvline(crash_time,color='grey',linestyle='--')
        axs[0].set_ylabel('ttc(model)/s')
        
        ttc_logic = data[:, 1]
        if ttc_logic[0] > 15:
            ttc_logic[0] = 15
        for i in range(1, len(ttc_logic)):
            if ttc_logic[i] > 15:
                ttc_logic[i] = ttc_logic[i-1] - 0.033

        axs[1].plot(time, data[:, 1])
        axs[1].set_ylabel('ttc(logic)/s')
        # axs[1].axvline(crash_time,color='grey',linestyle='--')

        axs[2].plot(time, data[:, 3])
        axs[2].set_xlabel('time/s')
        axs[2].set_ylabel('speed/kph')
        # axs[2].axvline(crash_time,color='grey',linestyle='--')

        plt.tight_layout()
        plt.savefig('output/fig/'+dlb_id+'.jpg', dpi=1000)
        plt.close()
