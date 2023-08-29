import cv2
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from visualize.base_render import BaseRender
from pyquaternion import Quaternion

# Define a constant for camera names
CAM_NAMES = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]


class CameraRender(BaseRender):
  """
    Render class for Camera View
    """

  def __init__(self, figsize=(53.3333, 20), show_gt_boxes=False):
    super().__init__(figsize)
    self.cams = CAM_NAMES
    self.show_gt_boxes = show_gt_boxes

  def get_axis(self, index):
    """Retrieve the corresponding axis based on the index."""
    if np.size(self.axes) == 1:
      return self.axes
    return self.axes[index // 3, index % 3]

  def project_to_cam(
      self,
      agent_prediction_list,
      sample_data_token,
      nusc,
      lidar_cs_record,
      project_traj=False,
      cam=None,
  ):
    """Project predictions to camera view."""
    _, cs_record, pose_record, cam_intrinsic, imsize = self.get_image_info(sample_data_token, nusc)
    boxes = []
    for agent in agent_prediction_list:
      box = Box(agent.pred_center,
                agent.pred_dim,
                Quaternion(axis=(0.0, 0.0, 1.0), radians=agent.pred_yaw),
                name=agent.pred_label,
                token='predicted')
      box.is_sdc = agent.is_sdc
      if project_traj:
        box.pred_traj = np.zeros((agent.pred_traj_max.shape[0] + 1, 3))
        box.pred_traj[:, 0] = agent.pred_center[0]
        box.pred_traj[:, 1] = agent.pred_center[1]
        box.pred_traj[:, 2] = agent.pred_center[2] - \
            agent.pred_dim[2]/2
        box.pred_traj[1:, :2] += agent.pred_traj_max[:, :2]
        box.pred_traj = (
            Quaternion(lidar_cs_record['rotation']).rotation_matrix @ box.pred_traj.T).T
        box.pred_traj += np.array(lidar_cs_record['translation'])[None, :]
      box.rotate(Quaternion(lidar_cs_record['rotation']))
      box.translate(np.array(lidar_cs_record['translation']))
      boxes.append(box)
    # Make list of Box objects including coord system transforms.

    box_list = []
    tr_id_list = []
    for i, box in enumerate(boxes):
      #  Move box to sensor coord system.
      box.translate(-np.array(cs_record['translation']))
      box.rotate(Quaternion(cs_record['rotation']).inverse)
      if project_traj:
        box.pred_traj += -np.array(cs_record['translation'])[None, :]
        box.pred_traj = (
            Quaternion(cs_record['rotation']).inverse.rotation_matrix @ box.pred_traj.T).T

      tr_id = agent_prediction_list[i].pred_track_id
      if box.is_sdc and cam == 'CAM_FRONT':
        box_list.append(box)
      if not box_in_image(box, cam_intrinsic, imsize):
        continue
      box_list.append(box)
      tr_id_list.append(tr_id)
    return box_list, tr_id_list, cam_intrinsic, imsize

  def render_image_data(self, sample_token, nusc):
    """Load and annotate image based on the provided path."""
    sample = nusc.get('sample', sample_token)
    # for i, cam in enumerate(self.cams):
    i, cam = 0, 'CAM_FRONT'
    sample_data_token = sample['data'][cam]
    data_path, _, _, _, _ = self.get_image_info(sample_data_token, nusc)
    image = self.load_image(data_path, cam)
    self.update_image(image, i, cam)

  def load_image(self, data_path, cam):
    """Update the axis of the plot with the provided image."""
    image = np.array(Image.open(data_path))
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 60)
    fontScale = 2
    color = (0, 0, 0)
    thickness = 4
    return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

  def update_image(self, image, index, cam):
    """Render image data for each camera."""
    ax = self.get_axis(index)
    ax.imshow(image)
    plt.axis('off')
    ax.axis('off')
    ax.grid(False)

  def get_image_info(self, sample_data_token, nusc):
    """Retrieve image information."""
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
      cam_intrinsic = np.array(cs_record['camera_intrinsic'])
      imsize = (sd_record['width'], sd_record['height'])
    else:
      cam_intrinsic = None
      imsize = None
    return data_path, cs_record, pose_record, cam_intrinsic, imsize
