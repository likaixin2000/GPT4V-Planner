from environment import Environment
import time
import numpy as np
import utils
import os
import pybullet as p

# IMPORTANT: set disp=False if you don't have a graphics interface
# Try running `sudo startx`, and connect with `ssh -X <dest>`
my_env = Environment(disp=True)  
my_env.reset()

# ------------------------------
# Environment Setup
# ------------------------------
# Add bowls.
bowl_size = (0.12, 0.12, 0)
n_bowls =1
bowl_urdf = 'bowl/bowl.urdf'
for _ in range(n_bowls):
    _,depth,_ = my_env.render(my_env.camera_config_up)
    bowl_pose = utils.get_random_pose(depth, bowl_size,my_env.pixel_size,my_env.camera_config_up)
    my_env.add_object(bowl_urdf, bowl_pose, 'fixed')

# Add blocks.
n_blocks = 1
block_size = (0.04, 0.04, 0.04)
block_urdf = 'stacking/block.urdf'
for _ in range(n_blocks):
    _,depth,_ = my_env.render(my_env.camera_config_up)
    block_pose = utils.get_random_pose(depth, block_size,my_env.pixel_size,my_env.camera_config_up)
    block_id = my_env.add_object(block_urdf, block_pose)


# Colors of distractor objects.
bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

n_distractors = 0
while n_distractors < 10:
    is_block = np.random.rand() > 0.5
    urdf = block_urdf if is_block else bowl_urdf
    size = block_size if is_block else bowl_size
    colors = block_colors if is_block else bowl_colors
    _,depth,_ = my_env.render(my_env.camera_config_up)
    pose = utils.get_random_pose(depth, size ,my_env.pixel_size,my_env.camera_config_up)
    if not pose[0] or not pose[1]:
        continue
    obj_id = my_env.add_object(urdf, pose)
    color = colors[n_distractors % len(colors)]
    p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
    n_distractors += 1

time.sleep(10)
#render
#step