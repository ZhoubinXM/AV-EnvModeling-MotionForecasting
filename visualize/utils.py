import cv2
import os
import math
import numpy as np
from pyquaternion import Quaternion


def combine(out_filename):
  # pass
  bev_image = cv2.imread(out_filename + '.jpg')
  cam_image = cv2.imread(out_filename + '_bev.jpg')
  merge_image = cv2.hconcat([cam_image, bev_image])
  cv2.imwrite(out_filename + '_combine.jpg', merge_image)
  # os.remove(out_filename + '_bev.jpg')


def obtain_map_info(nusc,
                    nusc_maps,
                    sample,
                    patch_size=(102.4, 102.4),
                    canvas_size=(256, 256),
                    layer_names=['lane_divider', 'road_divider'],
                    thickness=10):
  """
    Export 2d annotation from the info file and raw data.
    """
  l2e_r = sample['lidar2ego_rotation']
  l2e_t = sample['lidar2ego_translation']
  e2g_r = sample['ego2global_rotation']
  e2g_t = sample['ego2global_translation']
  l2e_r_mat = Quaternion(l2e_r).rotation_matrix
  e2g_r_mat = Quaternion(e2g_r).rotation_matrix

  scene = nusc.get('scene', sample['scene_token'])
  log = nusc.get('log', scene['log_token'])
  nusc_map = nusc_maps[log['location']]
  if layer_names is None:
    layer_names = nusc_map.non_geometric_layers

  l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
  l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
  patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
  patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])
  # TODO: render_map_mask() api
  map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size=canvas_size)
  map_mask = map_mask[-2] | map_mask[-1]
  map_mask = map_mask[np.newaxis, :]
  map_mask = map_mask.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)

  erode = nusc_map.get_map_mask(patch_box, patch_angle, ['drivable_area'], canvas_size=canvas_size)
  erode = erode.transpose((2, 1, 0)).squeeze(2)

  map_mask = np.concatenate([erode[None], map_mask[None]], axis=0)
  return map_mask
