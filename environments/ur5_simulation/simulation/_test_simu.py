from environment import Environment
import time
import numpy as np
import utils
import os

my_env = Environment(disp=True)
my_env.reset()
_,depth,_ = my_env.render(my_env.camera_config_up)
    # Add kit.
kit_size = (0.28, 0.2, 0.005)
kit_urdf = 'kitting/kit.urdf'
kit_pose = utils.get_random_pose(depth,kit_size,my_env.pixel_size,my_env.camera_config_up)
my_env.add_object(kit_urdf, kit_pose, 'fixed')

_,depth,_ = my_env.render(my_env.camera_config_up)
template = 'kitting/object-template.urdf'
obj_size = (0.08, 0.08, 0.02)
obj_pose = utils.get_random_pose(depth,obj_size,my_env.pixel_size,my_env.camera_config_up)

shape = os.path.join(my_env.assets_root, 'kitting','00.obj')
scale = [0.003, 0.003, 0.0001]  # .0005
replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
urdf = utils.fill_template(my_env.assets_root,template, replace)
my_env.add_object(urdf, obj_pose)
os.remove(urdf)

_,depth,_ = my_env.render(my_env.camera_config_up)
template = 'kitting/object-template.urdf'
obj_size = (0.08, 0.08, 0.02)
obj_pose = utils.get_random_pose(depth,obj_size,my_env.pixel_size,my_env.camera_config_up)

shape = os.path.join(my_env.assets_root, 'kitting','01.obj')
scale = [0.003, 0.003, 0.0001]  # .0005
replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
urdf = utils.fill_template(my_env.assets_root,template, replace)
my_env.add_object(urdf, obj_pose)
os.remove(urdf)

time.sleep(1)
# my_env.step([0.1,0.2],[0.5,0.5])
my_env.step_([120,160],[130,130])
# time.sleep(10)

# place_pixel = [0.2,0.5]
# pick_pos = obj_pose[0]
# pick_pixel = utils.position_to_pixel(pick_pos, my_env.camera_config_up, my_env.pixel_size)
# my_env.step(pick_pixel,place_pixel)


time.sleep(10)