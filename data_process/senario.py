from __future__ import annotations

import math
import copy
import numpy as np
from operator import attrgetter
import time
import matplotlib.pyplot as plt


class Point(object):

    def __init__(self, x=0, y=0, heading=0, ts=0) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.ts = ts

    def distance_to(self, other: Point) -> float:
        return math.sqrt(((self.x - other.x)**2) + ((self.y - other.y)**2))


class Obj(object):

    def __init__(self, id: int, point: Point, traj=None) -> None:
        self.id = id
        self.curr_pos = point
        if traj != None:
            self.traj = traj  # list of Point
        else:
            self.traj = [point]
        self.ts = point.ts
        self.pred_traj = []

    def insert(self, point: Point) -> None:
        self.traj.append(point)
        self.ts = point.ts
        self.curr_pos = point


class Lane(object):

    def __init__(self, start, end, c0, c1, c2, c3, step=1, expand=False) -> None:
        self.start = np.ceil(start).astype(np.int32)
        self.end = np.floor(end).astype(np.int32)
        # trick
        if expand:
            lane_len = (self.end - self.start)
            self.start -= np.floor(0.3 * lane_len).astype(np.int32)
            self.end += np.floor(0.7 * lane_len).astype(np.int32)

        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.step = int(step)
        self.lane_points = []
        self.generate_lane_point()

    def reset(self) -> None:
        self.start = -1
        self.end = -1
        self.c0 = -1
        self.c1 = -1
        self.c2 = -1
        self.c3 = -1
        self.lane_points = []

    def generate_lane_point(self) -> None:
        for x in range(self.start, self.end, self.step):
            y = self.c0 + self.c1 * x + self.c2 * math.pow(x, 2) + self.c3 * math.pow(x, 3)
            self.lane_points.append(Point(x, y))


class Senario(object):

    def __init__(self) -> None:
        self.reset()
        self.dis_threshold = 2
        self.fig, self.ax = plt.subplots()
        left, bottom, width, height = 0.2, 0.15, 0.6, 0.15
        self.sub_ax = self.fig.add_axes([left, bottom, width, height])

    def reset(self, dlb_uuid="") -> None:
        self.current_ids = []
        self.ego = Obj(0, Point(0, 0, 0))
        self.objs = {}
        self.lanes = []
        self.edges = []
        self.roads = []  # roads = lanes + edges
        self.centerlines = []
        self.discrete_ids = []
        self.snapshots = []
        self.expand_id = 1000
        self.best_len = 0
        self.best_snapshot = {}
        self.map_region = []
        self.dlb_uuid = dlb_uuid
        self.target_id = -1

    def trans_points(self, pos) -> Point:
        heading = math.atan2(pos.y, pos.x) + self.ego.curr_pos.heading
        dis = pos.distance_to(Point(0, 0))
        pos.x = self.ego.curr_pos.x + dis * math.cos(heading)
        pos.y = self.ego.curr_pos.y + dis * math.sin(heading)
        pos.heading += self.ego.curr_pos.heading
        return pos

    def update_lanes(self, lanes: list) -> None:
        self.lanes = lanes
        for lane in self.lanes:
            for point in lane.lane_points:
                point = self.trans_points(point)

    def update_edges(self, edges: list) -> None:
        self.edges = edges
        for edge in self.edges:
            for point in edge.lane_points:
                point = self.trans_points(point)

    # TODO: 路口场景暂不考虑
    def update_centerlines(self) -> None:
        # self.roads = self.lanes + self.edges
        self.roads = self.lanes
        self.roads.sort(key=attrgetter('c0'))
        self.centerlines = []
        for i in range(0, len(self.roads) - 1):
            start = min(self.roads[i].start, self.roads[i + 1].start)
            end = max(self.roads[i].end, self.roads[i + 1].end)
            c0 = 0.5 * (self.roads[i].c0 + self.roads[i + 1].c0)
            c1 = 0.5 * (self.roads[i].c1 + self.roads[i + 1].c1)
            c2 = 0.5 * (self.roads[i].c2 + self.roads[i + 1].c2)
            c3 = 0.5 * (self.roads[i].c3 + self.roads[i + 1].c3)
            centerline = Lane(start, end, c0, c1, c2, c3, expand=False)
            self.centerlines.append(centerline)
        for centerline in self.centerlines:
            for point in centerline.lane_points:
                point = self.trans_points(point)

    def generate_targets(self):
        pass

    def update_objs(self, current_time: int, points: dict) -> None:
        self.discrete_ids = []
        self.history_ids = copy.deepcopy(self.current_ids)
        for id, point in points.items():
            self.insert(id, point)
            # clear pred_tarj
            self.objs[id].pred_traj = []
        self.merge()
        self.drop_outdate_objs(current_time)
        # self.update_best_snapshot()

    def add_new_obj(self, id: int, point: Point) -> None:
        self.objs[id] = Obj(id, point)
        self.current_ids.append(id)
        self.discrete_ids.append(id)

    def insert(self, id: int, point: Point) -> None:
        if id in self.current_ids:
            curr_pos = self.objs[id].curr_pos
            if point.distance_to(curr_pos) < self.dis_threshold * 10:
                self.objs[id].insert(point)
                self.history_ids.remove(id)
            else:
                self.add_new_obj(self.expand_id, point)
                self.expand_id += 1
        else:
            self.add_new_obj(id, point)

    def merge(self) -> None:
        for discrete_id in self.discrete_ids:
            discrete_pos = self.objs[discrete_id].curr_pos
            for id in self.history_ids:
                end_point = self.objs[id].curr_pos
                if discrete_pos.distance_to(end_point) < self.dis_threshold:
                    self.objs[id].insert(discrete_pos)
                    self.objs.pop(discrete_id)
                    self.current_ids.remove(discrete_id)
                    break

    def drop_outdate_objs(self, current_time: int) -> None:
        outdate_id = []
        for id, obj in self.objs.items():
            if current_time - obj.ts > 0.5e9:
                outdate_id.append(id)
        for id in outdate_id:
            self.objs.pop(id)
            self.current_ids.remove(id)

    def save_snapshot(self) -> None:
        snapshot = {}
        snapshot["objs"] = copy.deepcopy(self.objs)
        snapshot["current_ids"] = copy.deepcopy(self.current_ids)
        snapshot["ego"] = copy.deepcopy(self.ego)
        snapshot["lanes"] = copy.deepcopy(self.lanes)
        snapshot["edges"] = copy.deepcopy(self.edges)
        snapshot["centerlines"] = copy.deepcopy(self.centerlines)
        self.snapshots.append(snapshot)

    # warning: many magic numbers
    def target_center_senario(self, obj: Obj) -> tuple:
        # assert target_id in self.current_ids
        assert len(obj.traj) >= 60
        target_obj = copy.deepcopy(obj)
        target_obj_traj = target_obj.traj[-1:0:-3]
        # target_obj_traj = target_obj_traj[:20]
        angle = self.cal_angle(target_obj_traj[::-1])
        curr_x, curr_y = target_obj_traj[0].x, target_obj_traj[0].y
        # for ego
        ego = copy.deepcopy(self.ego)
        # ego.traj = ego.traj[-1:0:-3]
        for p in ego.traj:
            p.x, p.y = self.rotate(p.x - curr_x, p.y - curr_y, angle)
        # for objs
        objs = {}
        for id in self.current_ids:
            obj = copy.deepcopy(self.objs[id])
            # obj.traj = obj.traj[-1:0:-3]
            for p in obj.traj:
                p.x, p.y = self.rotate(p.x - curr_x, p.y - curr_y, angle)
            objs[id] = obj
        # for centerlines
        centerlines = []
        for centerline in self.centerlines:
            line = copy.deepcopy(centerline)
            for p in line.lane_points:
                p.x, p.y = self.rotate(p.x - curr_x, p.y - curr_y, angle)
            centerlines.append(line)
        return ego, objs, centerlines

    # magic num: 30
    def update_target_pred_traj(self, target_id: int, predict_traj: np.array) -> None:
        '''
            predict_traj: [k,30,2]
        '''
        assert target_id in self.current_ids
        assert len(self.objs[target_id].traj) >= 60
        target_obj = copy.deepcopy(self.objs[target_id])
        target_obj_traj = target_obj.traj[-1:0:-3]
        angle = self.cal_angle(target_obj_traj[::-1])
        curr_x, curr_y = target_obj_traj[0].x, target_obj_traj[0].y
        self.objs[target_id].pred_traj = []
        for i in range(predict_traj.shape[0]):
            one_pred_traj = []
            for j in range(30):
                x, y = self.rotate(predict_traj[i, j, 0], predict_traj[i, j, 1], -angle)
                predict_traj[i, j, 0] = x + curr_x
                predict_traj[i, j, 1] = y + curr_y
                one_pred_traj.append(Point(x + curr_x, y + curr_y))
            self.objs[target_id].pred_traj.append(one_pred_traj)

    # magic num: 30
    def update_ego_pred_traj(self, predict_traj: np.array) -> None:
        '''
            predict_traj: [k,30,2]
        '''
        ego = copy.deepcopy(self.ego)
        ego_traj = ego.traj[-1:0:-3]
        angle = self.cal_angle(ego_traj[::-1])
        curr_x, curr_y = ego_traj[0].x, ego_traj[0].y
        self.ego.pred_traj = []
        for i in range(predict_traj.shape[0]):
            one_pred_traj = []
            for j in range(30):
                x, y = self.rotate(predict_traj[i, j, 0], predict_traj[i, j, 1], -angle)
                predict_traj[i, j, 0] = x + curr_x
                predict_traj[i, j, 1] = y + curr_y
                one_pred_traj.append(Point(x + curr_x, y + curr_y))
            self.ego.pred_traj.append(one_pred_traj)

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

    def rotate(self, x, y, angle):
        res_x = x * math.cos(angle) - y * math.sin(angle)
        res_y = x * math.sin(angle) + y * math.cos(angle)
        return res_x, res_y

    def check_crash(self) -> list:
        crash_id_time = {}
        for i in range(30):
            for ego_traj in self.ego.pred_traj:
                for id, obj in self.objs.items():
                    if id in crash_id_time.keys():
                        continue
                    for traj in obj.pred_traj:
                        if traj[i].distance_to(ego_traj[i]) <= 1:
                            ts = self.ego.ts + (i + 1) * 0.1 * 1e9
                            time_array = time.localtime(ts / 1e9)
                            time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
                            time_str += str(round(ts / 1e6) % 1000)
                            crash_id_time[id] = [i, ego_traj, traj, time_str, self.ego.ts, ts, ts - self.ego.ts]
                            break
        if crash_id_time != {}:
            time_array = time.localtime(self.ego.ts / 1e9)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
            time_str += str(round(self.ego.ts / 1e6) % 1000)
            # print(time_str, crash_id_time)
        self.crash_id_time = crash_id_time
        return crash_id_time

    def visualize(self, fig_path, show=False):
        self.ax.cla()

        #plot ego
        ego_pose = self.ego.curr_pos
        left_top_x = ego_pose.x + 2.5
        left_top_y = ego_pose.y + 1
        right_top_x = ego_pose.x + 2.5
        right_top_y = ego_pose.y - 1
        left_bottom_x = ego_pose.x - 2.5
        left_bottom_y = ego_pose.y + 1
        right_bottom_x = ego_pose.x - 2.5
        right_bottom_y = ego_pose.y - 1
        self.ax.plot([left_top_x, right_top_x, right_bottom_x, left_bottom_x, left_top_x],
                     [left_top_y, right_top_y, right_bottom_y, left_bottom_y, left_top_y],
                     '-',
                     lw=2,
                     c='r')
        x = [point.x for point in self.ego.traj]
        y = [point.y for point in self.ego.traj]
        self.ax.plot(x[-60:], y[-60:])
        for traj in self.ego.pred_traj:
            x = [point.x for point in traj]
            y = [point.y for point in traj]
            self.ax.plot(x, y, '-', lw=1, c='green')

        #plot objs: history traj and pred traj
        for obj_id, obj in self.objs.items():
            traj = obj.traj
            current_pos = traj[-1]
            self.ax.plot(current_pos.x, current_pos.y, 'o', lw=5, c='y')
            self.ax.text(current_pos.x, current_pos.y, str(obj_id))
            # history traj
            x = [point.x for point in traj]
            y = [point.y for point in traj]
            self.ax.plot(x[-60:], y[-60:], '-', lw=1, c='y')
            # pred traj
            for traj in obj.pred_traj:
                x = [point.x for point in traj]
                y = [point.y for point in traj]
                self.ax.plot(x, y, '-', lw=1, alpha=0.5, c='green')

        #plot lanes
        for lane in self.lanes:
            x = [point.x for point in lane.lane_points]
            y = [point.y for point in lane.lane_points]
            self.ax.plot(x, y, '-', lw=2, c='blue')

        #plot egdes
        for edge in self.edges:
            x = [point.x for point in edge.lane_points]
            y = [point.y for point in edge.lane_points]
            self.ax.plot(x, y, '-', lw=3, c='black')

        #plot centerlines
        for centerline in self.centerlines:
            x = [point.x for point in centerline.lane_points]
            y = [point.y for point in centerline.lane_points]
            self.ax.plot(x, y, linestyle=':', lw=1, c='grey')

        time_array = time.localtime(self.ego.ts / 1e9)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S\'", time_array)
        time_str += str(round(self.ego.ts / 1e6) % 1000)
        self.ax.set_title(self.dlb_uuid + "   " + time_str)
        self.ax.set_xlim(ego_pose.x - 75, ego_pose.x + 150)
        self.ax.set_ylim(ego_pose.y - 50, ego_pose.y + 50)

        self.sub_ax.cla()
        self.sub_ax.set_axis_off()
        for id, crash_info in self.crash_id_time.items():
            self.sub_ax.set_axis_on()
            x = [point.x for point in crash_info[2][:crash_info[0] + 1]]
            y = [point.y for point in crash_info[2][:crash_info[0] + 1]]
            self.sub_ax.text(crash_info[2][0].x, crash_info[2][0].y, str(id))
            self.sub_ax.plot(x, y, lw=1, c='green')
            x = [point.x for point in crash_info[1][:crash_info[0] + 1]]
            y = [point.y for point in crash_info[1][:crash_info[0] + 1]]
            self.sub_ax.text(crash_info[1][0].x, crash_info[1][0].y, "ego")
            self.sub_ax.plot(x, y, lw=1, c='blue')
            self.sub_ax.plot(crash_info[1][crash_info[0]].x, crash_info[1][crash_info[0]].y, 'o', lw=5, color="red")
            self.sub_ax.text(crash_info[1][crash_info[0]].x, crash_info[1][crash_info[0]].y, str(crash_info[0] / 10) + "s")
            self.sub_ax.set_ylim(crash_info[1][0].y - 5, crash_info[1][0].y + 5)
            self.sub_ax.set_xlim(crash_info[1][0].x - 30, crash_info[1][0].x + 75)
            self.sub_ax.set_title("crash possibility")
        if show:
            plt.pause(0.001)
        else:
            plt.savefig(fig_path + self.dlb_uuid + ".jpg")
