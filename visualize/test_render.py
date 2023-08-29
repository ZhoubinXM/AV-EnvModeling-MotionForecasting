import pickle
from camera_render import CameraRender, CAM_NAMES
from bev_render import BEVRender
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import combine

camera_render = CameraRender()
camera_render.reset_canvas(dx=2, dy=3, tight_layout=True)

with open('./data/nuscenes/trainval/infos/nuscenes_v1.0-mini_infos_temporal_train.pkl', 'rb') as f:
  train_set = pickle.load(f)

train_info = train_set['infos']

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
# from datasets.nuscenes_dataset import NuscenesDataset

ann_file = './data/nuscenes/infos/nuscenes_v1.0-mini_infos_temporal_train.pkl'
nusc = NuScenes(version='v1.0-mini', dataroot="./data/nuscenes/trainval/mini", verbose=False)
predict_helper = PredictHelper(nusc)
# nusc_dataset = NuscenesDataset(ann_file=ann_file,
#                                past_length=6,
#                                fut_length=12,
#                                data_root="./data/nuscenes/trainval/mini")
dataroot = "./data/nuscenes/trainval"
nusc_maps = {
    'boston-seaport': NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
    'singapore-hollandvillage': NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage'),
    'singapore-onenorth': NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
    'singapore-queenstown': NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown'),
}
# data_info = nusc_dataset.data_infos[18]
# sample_token = data_info['frame_token']
# camera_render.render_image_data(data_info['frame_token'], nusc)
# camera_render.render_dataset_frame(data_info, "./data/nuscenes/trainval/mini", only_fw=True)
# camera_render.save_fig("./dataset_1.jpg")

# camera_render_2 = CameraRender()
# for camera in CAM_NAMES:
#   camera_render_2.reset_canvas(dx=2, dy=2,tight_layout=True)
#   image_path = train_info[0]['cams'][camera][0]['image_filename']
#   import os
#   data_path = os.path.join("./data/nuscenes/trainval/mini", image_path)
#   image = camera_render_2.load_image(data_path, camera)
#   camera_render_2.update_image(image, 0, camera)
#   camera_render_2.save_fig("test_{}.jpg".format(camera))

bev_render = BEVRender()
# bev_render.reset_canvas(dx=1, dy=1)
# bev_render.set_plot_cfg()
# bev_render.render_dataset_frame(nusc_dataset.data_infos[18], only_fw=True)
# bev_render.render_dataset_frame(nusc_dataset.data_infos[18], only_fw=False)

# bev_render.render_anno_data(sample_token, nusc, predict_helper)
# bev_render.render_legend()
# bev_render.render_sdc_car()
# bev_render.render_hd_map(nusc, nusc_maps, sample_token)
# bev_render.save_fig('dataset_1_bev.jpg')
# bev_render.save_fig('test_bev_fw.jpg')

# combine('dataset_1')

# Test render predict res
bev_render.reset_canvas()
bev_render.set_plot_cfg()
bev_render.render_predict_trajs(sample_token,
                                pred_traj=np.array([[i, i] for i in range(20)]),
                                nusc=nusc,
                                prediction_helper=predict_helper,
                                nusc_maps=nusc_maps)
bev_render.save_fig("test_pred.jpg")
