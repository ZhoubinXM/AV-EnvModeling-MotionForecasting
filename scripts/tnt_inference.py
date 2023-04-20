from data_process.senario import *
from data_process.data_processor import *
import torch
import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from random import sample


class TNTModel(object):

    def __init__(self, filename) -> None:
        self.filename = filename
        self.data_processor = DataPorcessor()
        self.df = None
        self.hidden_size = 128
        self.use_semantic_lane = False
        self.max_target_num = 2000
        self.model_path = "/home/guifeng.fan/project/vectornet/saved_model/pth/vectornet_20220211_193210_epoch60.pth"

    def get_df(self):
        self.df = pd.read_csv(self.filename, low_memory=False)
        for i in range(0, 4):
            self.df = self.df.rename(
                # columns={
                #     "l_line_C0_" + str(i) + ".1": "l_line_C0_1" + str(i),
                #     "l_line_C1_" + str(i) + ".1": "l_line_C1_1" + str(i),
                #     "l_line_C2_" + str(i) + ".1": "l_line_C2_1" + str(i),
                #     "l_line_C3_" + str(i) + ".1": "l_line_C3_1" + str(i)
                # })
                columns={
                    "l_RE_line_C0_" + str(i): "l_line_C0_1" + str(i),
                    "l_RE_line_C1_" + str(i): "l_line_C1_1" + str(i),
                    "l_RE_line_C2_" + str(i): "l_line_C2_1" + str(i),
                    "l_RE_line_C3_" + str(i): "l_line_C3_1" + str(i)
                })
        if "l_age_128" not in self.df.columns:
            print("file invalid!")
            return

    def inference(self, visualize=False, res_csv="", fig_path=""):
        self.get_df()
        if self.df is None:
            print("None data frame!")
            return
        model = torch.load(self.model_path, map_location=torch.device('cuda:0'))
        model.eval()
        model.set_mode("inference")
        model = model.to("cuda")
        # tmp
        crash_count = {}
        res = [-1, -1, -1, -1]
        risk = False

        for row in self.df.itertuples():
            dlb_uuid = row.c_dlb_uuid
            self.data_processor.process(row)
            if row.l_VehSpdkph < 5:
                continue

            # for ego:
            self.ego_inference(model)

            # for objs
            self.obj_inference(model)

            # check crash
            crash_id_time = self.data_processor.senario.check_crash()

            # post-process
            for id, time in crash_id_time.items():
                if id not in crash_count.keys():
                    crash_count[id] = [1, time[4], time[5], time[6]]
                else:
                    crash_count[id] = [crash_count[id][0] + 1, time[4], time[5], time[6]]
            for id in crash_count.keys():
                if id not in crash_id_time.keys():
                    crash_count[id][0] = 0
            for id in crash_count.keys():
                if crash_count[id][0] >= 5:
                    res = crash_count[id]
                    risk = True
                    self.data_processor.senario.visualize(fig_path)
            # if risk:
            #     break
            # self.data_processor.senario.visualize(fig_path)


        # check result
        df_label = pd.read_csv("/data/jerome.zhou/adms_result_with_tag_0310.csv", low_memory=False)
        df_label = df_label[df_label['uuid'] == dlb_uuid]
        df_label = df_label.reset_index(drop=True)
        df_result = pd.read_csv(res_csv, low_memory=False)
        df_result.set_index(['dlb_uuid'], inplace=True)
        if len(df_label) == 1:
            res = pd.DataFrame(
                data={
                    'dlb_uuid': [dlb_uuid],
                    'current_ptp_ts': [res[1]],
                    'risk': [risk],
                    'predict_crash_time': [res[3] / 1e9],
                    'time_before_crash': [(df_label.at[0, 'crash_ptp_ts'] - res[1]) / 1e9],
                    'diff_of_label': [(res[2] - df_label.at[0, 'crash_ptp_ts']) / 1e9],
                    'crashclassification': [df_label.at[0, 'crashclassification']],
                    'lane_tag_normal': [df_label.at[0, 'lane_tag_normal']],
                    'lane_tag_abnormal': [df_label.at[0, 'lane_tag_abnormal']]
                })
            res.set_index(['dlb_uuid'], inplace=True)
            df_result = pd.concat([df_result, res])
        else:
            print("NO LABEL!")
        df_result.to_csv(res_csv)

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

    def genegrate_map_vector(
        self,
        centerline_frame: list,
        polylines: np.array,
        polyline_lens: np.array,
        id: int,
    ) -> None:
        polyline_lens[id] = len(centerline_frame) - 1
        for j, point in enumerate(centerline_frame):
            x, y = point.x, point.y
            if j > 0:
                vector = [0] * self.hidden_size
                vector[-1], vector[-2] = point_pre[0], point_pre[1]
                vector[-3], vector[-4] = x, y
                vector[-5] = 1
                vector[-6] = j
                vector[-7] = id
                if self.use_semantic_lane:
                    vector[-8] = -1  # has_traffic_control
                    vector[-9] = 0  # turn_direction
                    vector[-10] = -1  # intersection: 1 ; not intersection: -1
                point_pre_pre = (2 * point_pre[0] - x, 2 * point_pre[1] - y)
                if j >= 2:
                    point_pre_pre = (centerline_frame[j - 2].x, centerline_frame[j - 2].y)
                vector[-17] = point_pre_pre[0]
                vector[-18] = point_pre_pre[1]
                polylines[id, j - 1, :] = vector
            point_pre = [x, y]

    def lane_candidate_sampling(self, centerline_list, distance=0.5):
        """the input are list of lines, each line containing"""
        candidates = []
        for line in centerline_list:
            for i in range(len(line) - 1):
                # if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i + 1])):
                #     continue
                [x_diff, y_diff] = [line[i + 1].x - line[i].x, line[i + 1].y - line[i].y]
                if x_diff == 0.0 and y_diff == 0.0:
                    continue
                candidates.append([line[i].x, line[i].y])

                # compute displacement along each coordinate
                den = np.hypot(x_diff, y_diff) + np.finfo(float).eps
                d_x = distance * (x_diff / den)
                d_y = distance * (y_diff / den)

                num_c = np.floor(den / distance).astype(np.int32)
                # pt = copy.deepcopy(line[i])
                pt = np.array([line[i].x, line[i].y])
                for _ in range(num_c):
                    pt += np.array([d_x, d_y])
                    candidates.append(copy.deepcopy(pt))
        candidates = np.unique(np.asarray(candidates), axis=0)

        candidate_mask = np.zeros((self.max_target_num, 1), dtype=int)
        # keep candidates shape (self.max_target_num, 2)
        # random sample
        if (candidates.shape[0] >= self.max_target_num):
            index = sample(range(candidates.shape[0]), self.max_target_num)
            # candidates = candidates[:self.max_target_num, :]
            candidates = candidates[index, :]
        else:
            candidate_mask[candidates.shape[0]:] = -10000
            if candidates.size > 0:
                candidates = np.pad(candidates, ((0, self.max_target_num - candidates.shape[0]), (0, 0)), 'constant')
            else:
                candidates = np.zeros((self.max_target_num, 2), dtype=int)
        candidates = np.expand_dims(candidates, axis=0)
        candidate_mask = np.expand_dims(candidate_mask, axis=0)

        return candidates, candidate_mask

    def cal_angle(self, traj):
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

    def cal_speed(self, traj):
        duration = len(traj) * 0.1 - 0.1
        distance = np.sqrt((traj[-1].x - traj[0].x)**2 + (traj[-1].y - traj[0].y)**2)
        speed = distance / duration
        return speed

    def cal_ploy_num(self, senario):
        poly_num_lanes = 0
        for lane in senario.centerlines:
            frame = (len(lane.lane_points) + 19) // 20
            poly_num_lanes += frame
        # obj + lane + ego
        return len(senario.current_ids) + poly_num_lanes + 1

    def rotate(self, x, y, angle):
        res_x = x * math.cos(angle) - y * math.sin(angle)
        res_y = x * math.sin(angle) + y * math.cos(angle)
        return res_x, res_y

    def get_pad_vector(self, li, hidden_size):
        """
        Pad vector to length of hidden_size
        """
        assert len(li) <= hidden_size
        li.extend([0] * (hidden_size - len(li)))
        return li

    def split_centerline(self, centerline):
        centerline_frames = []
        centerline_len = len(centerline.lane_points)
        frame = (centerline_len + 19) // 20
        point_per_frame = (centerline_len + frame - 1) // frame
        for i in range(frame):
            if i == frame - 1:
                centerline_frames.append(centerline.lane_points[i * point_per_frame:])
            else:
                centerline_frames.append(centerline.lane_points[i * point_per_frame:(i + 1) * point_per_frame])
        return centerline_frames

    def select_traj(self, traj_in, score, threshold=2):
        """
        select the top k trajectories according to the score and the distance
        :param traj_in: candidate trajectories, [batch, M, horizon * 2]
        :param score: score of the candidate trajectories, [batch, M]
        :param threshold: float, the threshold for exclude traj prediction
        :return: [batch_size, k, horizon * 2]
        """

        def distance_metric(traj_candidate, traj_gt):
            if traj_candidate.dim() == 2:
                traj_candidate = traj_candidate.unsqueeze(1)
            _, M, horizon_2_times = traj_candidate.size()
            dis = torch.pow(traj_candidate - traj_gt, 2).view(-1, M, int(horizon_2_times / 2), 2)
            dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)
            return dis

        # re-arrange trajectories according the the descending order of the score
        M, k = 50, 6
        _, batch_order = score.sort(descending=True)
        traj_pred = torch.cat([traj_in[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, M, 30 * 2)
        traj_selected = traj_pred[:, :k]  # [batch_size, 1, horizon * 2]
        for batch_id in range(traj_pred.shape[0]):  # one batch for a time
            traj_cnt = 1
            for i in range(1, M):
                dis = distance_metric(traj_selected[batch_id, :traj_cnt, :], traj_pred[batch_id, i].unsqueeze(0))
                if not torch.any(dis < threshold * threshold):  # not exist similar trajectory
                    traj_selected[batch_id, traj_cnt] = traj_pred[batch_id, i]  # add this trajectory
                    traj_cnt += 1

                if traj_cnt >= k:
                    break  # break if collect enough traj

            # no enough traj, pad zero traj
            if traj_cnt < k:
                traj_selected[:, traj_cnt:] = 0.0
        return traj_selected

    def ego_inference(self, model):
        ego = self.data_processor.senario.ego
        if len(ego.traj) < 60:
            return
        ego, objs, centerlines = self.data_processor.senario.target_center_senario(ego)
        poly_num = self.cal_ploy_num(self.data_processor.senario)
        polylines = np.zeros((poly_num, 19, self.hidden_size), dtype=float)
        polyline_lens = np.zeros(poly_num, dtype=int)
        # objs, including ego and target
        self.genegrate_obj_vector(ego, polylines, polyline_lens, 0, 1)
        polyline_id = 1
        for id in self.data_processor.senario.current_ids:
            self.genegrate_obj_vector(objs[id], polylines, polyline_lens, polyline_id, 2)
            polyline_id += 1

        # centerlines
        centerline_list = []
        for lane in centerlines:
            centerline = copy.deepcopy(lane)
            # trick: 如果lane的points数目大于20，平均切分
            centerline_frames = self.split_centerline(centerline)
            centerline_list += centerline_frames
            for centerline_frame in centerline_frames:
                self.genegrate_map_vector(centerline_frame, polylines, polyline_lens, polyline_id)
                polyline_id += 1

        # mask
        mask = np.zeros([polylines.shape[0], polylines.shape[1], self.hidden_size // 2])
        for i, polyline_len in enumerate(polyline_lens):
            mask_len = polylines.shape[1] - polyline_len
            if mask_len > 0:
                mask[i, polyline_len:, :] = np.ones([mask_len, self.hidden_size // 2]) * (-10000.0)

        # candidates
        candidates, candidate_mask = self.lane_candidate_sampling(centerline_list, distance=1)

        # inference
        polylines = torch.from_numpy(polylines).float()
        poly_num = torch.from_numpy(np.array([polylines.shape[0]])).int()
        mask = torch.from_numpy(mask).int()
        candidates = torch.from_numpy(candidates).float()
        candidate_mask = torch.from_numpy(candidate_mask).int()

        predict_traj, predict_prob = model(polylines.to("cuda"), poly_num.to("cuda"), mask.to("cuda"), candidates.to("cuda"),
                                           candidate_mask.to("cuda"))
        traj_select = self.select_traj(predict_traj, predict_prob, 2)
        predict_traj = traj_select.to("cpu")
        predict_traj = predict_traj.reshape(-1, 30, 2).detach().numpy()
        self.data_processor.senario.update_ego_pred_traj(predict_traj)

    def obj_inference(self, model):
        for target_id, obj in self.data_processor.senario.objs.items():
            if len(obj.traj) < 60:
                continue
            ego, objs, centerlines = self.data_processor.senario.target_center_senario(obj)

            # 过滤车道线外的目标
            has_left, has_right = False, False
            for centerline in centerlines:
                for p in centerline.lane_points:
                    if p.x < 1 and p.y > 20:
                        has_left = True
                    if p.x > -1 and p.y > 20:
                        has_right = True
            if not has_left or not has_right:
                continue

            # downsample traj 30hz -> 10hz
            traj = obj.traj[-1:0:-3]
            traj = traj[:20]
            # angel = self.cal_angle(traj[::-1])
            if self.cal_speed(traj) < 5:
                continue

            poly_num = self.cal_ploy_num(self.data_processor.senario)
            polylines = np.zeros((poly_num, 19, self.hidden_size), dtype=float)
            polyline_lens = np.zeros(poly_num, dtype=int)

            # objs, including ego and target
            self.genegrate_obj_vector(objs[target_id], polylines, polyline_lens, 0, 1)
            self.genegrate_obj_vector(ego, polylines, polyline_lens, 1, 0)
            polyline_id = 2
            for id in self.data_processor.senario.current_ids:
                if id != target_id:
                    self.genegrate_obj_vector(objs[id], polylines, polyline_lens, polyline_id, 2)
                    polyline_id += 1

            # centerlines
            centerline_list = []
            for lane in centerlines:
                centerline = copy.deepcopy(lane)
                # trick: 如果lane的points数目大于20，平均切分
                centerline_frames = self.split_centerline(centerline)
                centerline_list += centerline_frames
                for centerline_frame in centerline_frames:
                    self.genegrate_map_vector(centerline_frame, polylines, polyline_lens, polyline_id)
                    polyline_id += 1

            # mask
            mask = np.zeros([polylines.shape[0], polylines.shape[1], self.hidden_size // 2])
            for i, polyline_len in enumerate(polyline_lens):
                mask_len = polylines.shape[1] - polyline_len
                if mask_len > 0:
                    mask[i, polyline_len:, :] = np.ones([mask_len, self.hidden_size // 2]) * (-10000.0)

            # candidates
            candidates, candidate_mask = self.lane_candidate_sampling(centerline_list, distance=1)

            # inference
            polylines = torch.from_numpy(polylines).float()
            poly_num = torch.from_numpy(np.array([polylines.shape[0]])).int()
            mask = torch.from_numpy(mask).int()
            candidates = torch.from_numpy(candidates).float()
            candidate_mask = torch.from_numpy(candidate_mask).int()

            predict_traj, predict_prob = model(polylines.to("cuda"), poly_num.to("cuda"), mask.to("cuda"),
                                               candidates.to("cuda"), candidate_mask.to("cuda"))
            traj_select = self.select_traj(predict_traj, predict_prob, 2)
            predict_traj = traj_select.to("cpu")
            predict_traj = predict_traj.reshape(-1, 30, 2).detach().numpy()
            self.data_processor.senario.update_target_pred_traj(target_id, predict_traj)
            # if target_id == 95:
            # self.target_center_visualize(ego, objs, centerlines, predict_traj, candidates)
            # print(predict_prob)

    def target_center_visualize(self, ego, objs, centerlines, pred_traj, candidates):
        plt.cla()

        #plot ego
        current_pos = ego.traj[-1]
        plt.plot(current_pos.x, current_pos.y, 'o', lw=5, c='r')
        x = [point.x for point in ego.traj]
        y = [point.y for point in ego.traj]
        plt.plot(x, y)

        #plot objs
        for obj_id, obj in objs.items():
            traj = obj.traj
            current_pos = traj[-1]
            plt.plot(current_pos.x, current_pos.y, 'o', lw=5, c='y')
            plt.text(current_pos.x, current_pos.y, str(obj_id))
            x = [point.x for point in traj]
            y = [point.y for point in traj]
            plt.plot(x, y, '-', lw=1, c='y')

        #plot centerlines
        for centerline in centerlines:
            x = [point.x for point in centerline.lane_points]
            y = [point.y for point in centerline.lane_points]
            plt.plot(x, y, linestyle=':', lw=1, c='grey')

        #plot pred_traj
        for i in range(pred_traj.shape[0]):
            plt.plot(pred_traj[i, :, 0], pred_traj[i, :, 1], lw=1, c='blue')

        #plot candidates
        for i in range(candidates.shape[1]):
            plt.plot(candidates[0, i, 0], candidates[0, i, 1], '*', lw=1, c='green')

        time_array = time.localtime(ego.ts / 1e9)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
        time_str += str(round(ego.ts / 1e6) % 1000)
        plt.title(time_str)
        plt.xlim(-20, 20)
        plt.ylim(-200, 200)
        plt.pause(0.033)


if __name__ == "__main__":
    # filename = "/data/nio_dataset/accident/database/db_accident_20230112_prod_15.csv"
    # filename = "/data/jerome.zhou/database/accident/normal_db_cutin_adms_result_with_tag_0310.csv"
    # filename = "/data/jerome.zhou/database/accident/risk_db_cutin_adms_result_with_tag_0310_2000.csv"
    df = pd.DataFrame(
        columns=['dlb_uuid', 'current_ptp_ts', 'risk', 'predict_crash_time', 'time_before_crash', 'diff_of_label'])
    df.set_index(['dlb_uuid'], inplace=True)
    tnt = TNTModel("filename")

    # # collision
    # root = "/data/guifeng/crash_dataset/collision"
    # res_csv = "/data/guifeng/crash_dataset/result/collision.csv"
    # fig_path="/data/guifeng/crash_dataset/fig/collision/"
    # df.to_csv(res_csv)
    # for _, _, files in os.walk(root):
    #     for file in files:
    #         filename = os.path.join(root, file)
    #         print("deal with: " + filename)
    #         tnt.filename = filename
    #         tnt.inference(visualize=True, res_csv=res_csv, fig_path=fig_path)
    
    # # filename = os.path.join(root, "8010dd3f-87f2-4e86-bde2-84b75d92f0cf.csv")
    # filename = "/data/nio_dataset/accident/database/db_accident_20230112_prod_15.csv"
    # tnt.filename = filename
    # tnt.inference(visualize=True, res_csv=res_csv, fig_path=fig_path)
    
    # # risk
    # root = "/data/guifeng/crash_dataset/risk"
    # res_csv = "/data/guifeng/crash_dataset/result/risk.csv"
    # fig_path="/data/guifeng/crash_dataset/fig/risk/"
    # df.to_csv(res_csv)
    # for _, _, files in os.walk(root):
    #     for file in files:
    #         filename = os.path.join(root, file)
    #         print("deal with: " + filename)
    #         tnt.filename = filename
    #         tnt.inference(visualize=True, res_csv=res_csv, fig_path=fig_path)

    # normal
    root = "/data/guifeng/crash_dataset/normal"
    res_csv = "/data/guifeng/crash_dataset/result/normal.csv"
    fig_path="/data/guifeng/crash_dataset/fig/normal/"
    df.to_csv(res_csv)
    for _, _, files in os.walk(root):
        for file in files:
            filename = os.path.join(root, file)
            print("deal with: " + filename)
            tnt.filename = filename
            tnt.inference(visualize=True, res_csv=res_csv, fig_path=fig_path)

