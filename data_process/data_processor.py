from data_process.senario import *
from data_process.sample_generator import *
import pandas as pd
import data_process.utils as utils
# import matplotlib.pyplot as plt
# import time


class DataPorcessor(object):

    def __init__(self, config={}) -> None:
        self.senario = Senario()
        self.sample_generator = SampleGenerator(config)
        self.config = config
        self.reset()

    def reset(self, dlb_uuid=None, dlb_start_time=-1):
        self.ptp_ts = -1
        self.dlb_uuid = dlb_uuid
        self.dlb_start_time = dlb_start_time
        self.dlb_snapshot_time = 1e9  # ns
        self.senario.reset(dlb_uuid)

    def process(self, row):
        # check dlb
        dlb_uuid = row.c_dlb_uuid
        if dlb_uuid != self.dlb_uuid:
            if self.dlb_uuid != None and self.senario.snapshots != []:
                self.generate_sample()
            self.reset(dlb_uuid=dlb_uuid, dlb_start_time=row.l_dms_ptp)
        self.dlb_uuid = dlb_uuid
        # get ptp
        self.ptp_ts = row.l_dms_ptp
        # # data before trigger time
        # if row.l_trigger_publish_ptp_ts <= self.ptp_ts:
        #     return
        # deal with ego: must be first
        self.process_ego(row)
        # deal with objs
        self.process_objs(row)
        # deal with road
        self.process_roads(row)
        # snapshot ever 1s
        if self.ptp_ts - self.dlb_start_time > self.dlb_snapshot_time:
            self.senario.save_snapshot()
            self.dlb_snapshot_time += 1e9

    def generate_sample(self) -> None:
        # deal with snapshot
        snapshot = self.generate_best_snapshot()
        if snapshot == {}:
            return
        snapshot = self.target_center_snapshot(snapshot)
        self.save_snapshot_fig(snapshot)
        # self.sample_generator.generate_sample(snapshot)

    def generate_best_snapshot(self) -> dict:
        snapshots = self.senario.snapshots
        target_id = -1
        for i, snapshot in enumerate(snapshots):
            objs = snapshot['objs']
            for id, obj in objs.items():
                if id != 0 and len(obj.traj) > (self.config['history_points'] + self.config['prediction_points'] +
                                                self.config['extra_points']):
                    target_id = id
                    traj_gt = obj.traj
                    break
            if target_id != -1:
                break
        if target_id == -1:
            return {}
        snapshot_id = -1
        for i, snapshot in enumerate(snapshots):
            if target_id not in snapshot['current_ids']:
                continue
            obj = snapshot['objs'][target_id]
            if len(obj.traj) > self.config['history_points']:
                snapshot_id = i
                his_traj_len = len(obj.traj)
                break
        if snapshot_id == -1:
            return {}
        snapshot = snapshots[snapshot_id]
        snapshot['traj_gt'] = traj_gt[his_traj_len + 1:]
        snapshot['target_id'] = target_id
        return snapshot

    def target_center_snapshot(self, snapshot: dict) -> tuple:
        target_id = snapshot['target_id']
        target_obj = copy.deepcopy(snapshot['objs'][target_id])
        target_obj_traj = target_obj.traj[::-1]
        angle = utils.cal_angle(target_obj_traj[::-1])
        curr_x, curr_y = target_obj_traj[0].x, target_obj_traj[0].y
        # for ego
        ego = snapshot['ego']
        for p in ego.traj:
            p.x, p.y = utils.rotate(p.x - curr_x, p.y - curr_y, angle)
        # for objs
        for obj in snapshot['objs'].values():
            for p in obj.traj:
                p.x, p.y = utils.rotate(p.x - curr_x, p.y - curr_y, angle)
        # for centerlines
        for centerline in snapshot['centerlines']:
            for p in centerline.lane_points:
                p.x, p.y = utils.rotate(p.x - curr_x, p.y - curr_y, angle)
        # for lanes
        for lane in snapshot['lanes']:
            for p in lane.lane_points:
                p.x, p.y = utils.rotate(p.x - curr_x, p.y - curr_y, angle)
        # for edges
        for edge in snapshot['edges']:
            for p in edge.lane_points:
                p.x, p.y = utils.rotate(p.x - curr_x, p.y - curr_y, angle)
        # for traj_gt
        for p in snapshot['traj_gt']:
            p.x, p.y = utils.rotate(p.x - curr_x, p.y - curr_y, angle)
        return snapshot

    def save_snapshot_fig(self, snapshot):
        fig, ax = plt.subplots()
        plt.cla()
        ego, objs, centerlines = snapshot['ego'], snapshot['objs'], snapshot['centerlines']
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
            plt.text(current_pos.x, current_pos.y, str(int(obj_id)))
            x = [point.x for point in traj]
            y = [point.y for point in traj]
            plt.plot(x, y, '-', lw=1, c='y')

        # plot traj_gt
        traj = snapshot['traj_gt']
        current_pos = traj[-1]
        x = [point.x for point in traj]
        y = [point.y for point in traj]
        plt.plot(x, y, '-', lw=1, c='green')

        #plot lanes
        for lane in snapshot['lanes']:
            x = [point.x for point in lane.lane_points]
            y = [point.y for point in lane.lane_points]
            plt.plot(x, y, '-', lw=2, c='blue')

        #plot edges
        for edge in snapshot['edges']:
            x = [point.x for point in edge.lane_points]
            y = [point.y for point in edge.lane_points]
            plt.plot(x, y, '-', lw=3, c='black')

        #plot centerlines
        for centerline in centerlines:
            x = [point.x for point in centerline.lane_points]
            y = [point.y for point in centerline.lane_points]
            plt.plot(x, y, linestyle=':', lw=1, c='grey')

        time_array = time.localtime(ego.ts / 1e9)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
        time_str += str(round(ego.ts / 1e6) % 1000)
        plt.title(time_str)
        # plt.xlim(-20, 20)
        # plt.ylim(-200, 200)
        plt.savefig(os.path.join(self.config['dataset']['snapshotpath'], self.dlb_uuid + ".jpg"))
        plt.close()

    def process_ego(self, row):
        ego_feature = self.get_ego_feature(row)
        speed = ego_feature["VehSpdkph"]
        yaw_rate = ego_feature["YawRateSAERps"]
        if self.senario.ego.ts == 0:
            self.senario.ego = Obj(0, Point(0, 0, 0, self.ptp_ts))
        else:
            curr_point = Point(0, 0, 0, self.ptp_ts)
            ego = self.senario.ego
            time_scope = (self.ptp_ts - ego.ts) / 1e9
            curr_point.heading = ego.curr_pos.heading + time_scope * yaw_rate
            dis = speed * time_scope / 3.6
            curr_point.x = ego.curr_pos.x + dis * math.cos(curr_point.heading)
            curr_point.y = ego.curr_pos.y + dis * math.sin(curr_point.heading)
            ego.insert(curr_point)

    def process_objs(self, row):
        points = {}
        for i in range(1, 129):
            feature = self.get_obj_feature(row, i)
            obj_id = feature['id']
            if pd.isna(obj_id) or obj_id == -1:
                continue
            if feature['fusion_source'] not in [4, 5, 6, 7] or feature['main_class'] not in [1, 2]:
                continue
            pos = Point(feature['pos_x'], feature['pos_y'], feature['heading'], self.ptp_ts)
            points[obj_id] = self.senario.trans_points(pos)
        self.senario.update_objs(self.ptp_ts, points)

    def process_roads(self, row):
        lanes = []
        for i in range(10):
            lane_feature = self.get_lane_feature(row, i)
            if pd.notna(lane_feature['c0']):
                lane = Lane(start=lane_feature['start'],
                            end=lane_feature['end'],
                            c0=lane_feature['c0'],
                            c1=lane_feature['c1'],
                            c2=lane_feature['c2'],
                            c3=lane_feature['c3'])
                lanes.append(lane)
        self.senario.update_lanes(lanes)
        edges = []
        for i in range(10, 14):
            edge_feature = self.get_edge_feature(row, i)
            if pd.notna(edge_feature['c0']):
                edge = Lane(start=edge_feature['start'],
                            end=edge_feature['end'],
                            c0=edge_feature['c0'],
                            c1=edge_feature['c1'],
                            c2=edge_feature['c2'],
                            c3=edge_feature['c3'])
                edges.append(edge)
        self.senario.update_edges(edges)
        self.senario.update_centerlines()

    def get_obj_feature(self, row, idx):
        feature = {}
        feature['id'] = eval("row.c_obf_id_" + str(idx))
        feature['age'] = eval("row.l_age_" + str(idx))
        feature['pos_x'] = eval("row.l_pos_x_" + str(idx))
        feature['pos_y'] = -eval("row.l_pos_y_" + str(idx))
        feature['vx'] = eval("row.l_vx_" + str(idx))
        feature['vy'] = -eval("row.l_vy_" + str(idx))
        feature['ax'] = eval("row.l_ax_" + str(idx))
        feature['ay'] = -eval("row.l_ay_" + str(idx))
        feature['obj_sts'] = eval("row.c_obj_sts_" + str(idx))
        feature['heading'] = -eval("row.l_heading_" + str(idx))
        feature['length'] = eval("row.l_length_" + str(idx))
        feature['width'] = eval("row.l_width_" + str(idx))
        feature['main_class'] = eval("row.c_main_class_" + str(idx))
        feature['fusion_source'] = eval("row.c_fusion_source_" + str(idx))
        return feature

    def get_lane_feature(self, row, idx):
        feature = {}
        feature['c0'] = eval("row.l_line_C0_" + str(idx))
        feature['c1'] = eval("row.l_line_C1_" + str(idx))
        feature['c2'] = eval("row.l_line_C2_" + str(idx))
        feature['c3'] = eval("row.l_line_C3_" + str(idx))
        feature['type'] = eval("row.c_LD_Type_" + str(idx))
        feature['start'] = eval("row.l_LD_Start_" + str(idx))
        feature['end'] = eval("row.l_LD_End_" + str(idx))
        return feature

    def get_edge_feature(self, row, idx):
        feature = {}
        feature['c0'] = eval("row.l_line_C0_" + str(idx))
        feature['c1'] = eval("row.l_line_C1_" + str(idx))
        feature['c2'] = eval("row.l_line_C2_" + str(idx))
        feature['c3'] = eval("row.l_line_C3_" + str(idx))
        feature['type'] = eval("row.c_LD_RE_Type_" + str(idx - 10))
        feature['start'] = eval("row.l_LD_RE_VR_Start_" + str(idx - 10))
        feature['end'] = eval("row.l_LD_RE_VR_End_" + str(idx - 10))
        return feature

    def get_ego_feature(self, row):
        feature = {}
        feature['VehSpdkph'] = row.l_VehSpdkph
        feature['Trvl'] = row.l_Trvl
        feature['StrWhlAgSAE'] = row.l_StrWhlAgSAE
        feature['StrWhlAgSpdSAE'] = row.l_StrWhlAgSpdSAE
        feature['TorsBarTqSAE'] = row.l_TorsBarTqSAE
        feature['LgtSAEAg'] = row.l_LgtSAEAg
        feature['LgtSAEAmpss'] = row.l_LgtSAEAmpss
        feature['LatSAEAg'] = row.l_LatSAEAg
        feature['LatSAEAmpss'] = row.l_LatSAEAmpss
        feature['YawRateSAERps'] = -row.l_YawRateSAERps
        return feature


# test
if __name__ == "__main__":
    data_processor = DataPorcessor()
    data_processor.process(filename="/data/nio_dataset/accident/database/db_accident_20230112_prod_15.csv", visualize=True)
