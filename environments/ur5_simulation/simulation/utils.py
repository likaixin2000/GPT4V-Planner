import pybullet as p
import string
import tempfile
import os
import random
import numpy as np
import cv2 as cv
from transforms3d import euler


def load_urdf(pybullet_client, file_path, *args, **kwargs):
  """Loads the given URDF filepath."""
  # Handles most general file open case.
  try:
    return pybullet_client.loadURDF(file_path, *args, **kwargs)
  except pybullet_client.error:
    pass

def fill_template(assets_root, template, replace):
  """Read a file and replace key strings."""
  full_template_path = os.path.join(assets_root, template)
  with open(full_template_path, 'r') as file:
    fdata = file.read()
  for field in replace:
    for i in range(len(replace[field])):
      fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
  alphabet = string.ascii_lowercase + string.digits
  rname = ''.join(random.choices(alphabet, k=16))
  tmpdir = tempfile.gettempdir()
  template_filename = os.path.split(template)[-1]
  fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
  with open(fname, 'w') as file:
    file.write(fdata)
  return fname
  
def get_random_pose(depth, obj_size,pixel_size, camera_config):
  """Get random collision-free object pose within workspace bounds."""

  # Get erosion size of object in pixels.
  max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
  erode_size = int(np.round(max_size / pixel_size))

  # Randomly sample an object pose within free-space pixels.
  depth_threhold = 0.749
  free = np.ones(depth.shape, dtype=np.uint8)
  free[np.where(depth<depth_threhold)] = 0
  free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
  free = cv.erode(free, np.ones((erode_size, erode_size), np.uint8))

  if np.sum(free) == 0:
    return None, None
  pix = sample_distribution(np.float32(free))
  pos = pixel_to_position([pix[1],pix[0]], depth[pix[0]][pix[1]], camera_config, pixel_size)
  pos = (pos[0], pos[1], obj_size[2] / 2)
  theta = np.random.rand() * 2 * np.pi
  rot = eulerXYZ_to_quatXYZW((0, 0, theta))
  return pos, rot
  
def sample_distribution(prob, n_samples=1):
  """Sample data point from a custom distribution."""
  flat_prob = prob.flatten() / np.sum(prob)
  rand_ind = np.random.choice(
      np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
  rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
  return np.int32(rand_ind_coords.squeeze())

def pixel_to_position(pixel, height, camera_config,  pixel_size):
  """Convert from pixel  to world."""
  camera_pos = camera_config['position']
  image_width = camera_config['image_size'][1]
  image_height = camera_config['image_size'][0]
  u =  pixel[0]
  v =  pixel[1]
  x = camera_pos[0] + (v-image_height/2) * pixel_size
  y = camera_pos[1] + (u-image_width/2) * pixel_size
  z = camera_pos[2] - height
  if z < 0:
      z = 0
  return [x, y, z]


def position_to_pixel(position, camera_config, pixel_size):
  """Convert from 3D position to pixel location on heightmap."""
  camera_pos = camera_config['position']
  image_width = camera_config['image_size'][1]
  image_height = camera_config['image_size'][0]
  u = int(image_width/2 + (position[1] - camera_pos[1]) / pixel_size)
  v = int(image_height/2 + (position[0] - camera_pos[0]) / pixel_size)
  if u >= image_width:
      u = image_width-1
  if v >= image_height:
      v = image_height-1
  return [u, v]


def eulerXYZ_to_quatXYZW(rotation):  # pylint: disable=invalid-name
  euler_zxy = (rotation[2], rotation[0], rotation[1])
  quaternion_wxyz = euler.euler2quat(*euler_zxy, axes='szxy')
  q = quaternion_wxyz
  quaternion_xyzw = (q[1], q[2], q[3], q[0])
  return quaternion_xyzw


def quatXYZW_to_eulerXYZ(quaternion_xyzw):  # pylint: disable=invalid-name
  quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
  euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
  euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
  return euler_xyz


#-----------------------------------------------------------------------------
# PLOT UTILS
#-----------------------------------------------------------------------------

# Plot colors (Tableau palette).
COLORS = {'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
                   'red':  [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
                   'green':  [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
                  'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
                 'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
                 'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
                 'pink':   [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
                 'cyan':   [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
                'brown':  [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
                'gray':   [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0]}