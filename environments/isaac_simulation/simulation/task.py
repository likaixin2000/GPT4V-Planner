import os
from PIL import Image
from .environment import Environment
from . import simulation_utils as utils

class Task():
    def __init__(self, asset_root=None, enable_gui=False):
        if asset_root is None:
            self._env = Environment(asset_root=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "assets"), enable_gui=enable_gui)
        else:
            self._env = Environment(asset_root=asset_root, enable_gui=enable_gui)

        self.pixel_size = 1 / 1066 # when camera pose is at z=1 and image size is 1280*720

    # init object in the environment and set camera
    def reset(self):
        self._env.reset()
        '''
        add task specific reset here
        '''
        self._env.empty_step(60)


    def get_image(self):
        image_rgba=self._env.render()
        image = Image.fromarray(image_rgba[:,:,:3]).convert("RGB")
        return image
    
    # for pick and place tasks
    # todo here
    def step(self,pick_point,place_point):
        depth = self._env.get_depth()
        height, width = depth.shape
        pick_point_x, pick_point_y = pick_point
        place_point_x, place_point_y = place_point
        pick_point_x = int(pick_point_x * width)
        pick_point_y = int(pick_point_y * height)
        place_point_x = int(place_point_x * width)
        place_point_y = int(place_point_y * height)

        print(f"pick_point: {pick_point_x},{pick_point_y}")
        print(f"place_point: {place_point_x},{place_point_y}")
        camera_config={
            'position':[self._env.camera_pose.x,self._env.camera_pose.y,self._env.camera_pose.z],
            'image_size':[height,width]
        }
        pixel_size = self.pixel_size

        # 注意这里深度depth是负数，所以我们取反 而且是 720*1080
        pick_point_tf = utils.pixel_to_position(
            [pick_point_x,pick_point_y],
            -depth[pick_point_y][pick_point_x],
            camera_config,
            pixel_size
        )
        place_point_tf = utils.pixel_to_position(
            [place_point_x,place_point_y],
            -depth[place_point_y][place_point_x],
            camera_config,
            pixel_size
        )
        print(f"pick_point_tf: {pick_point_tf}")
        print(f"place_point_tf: {place_point_tf}")
        # todo
        # pick
        # place
        name=self._env.pick_place(pick_point_tf,place_point_tf)
        if not name:
            print("pick_place failed")
        else:
            print(f"pick_place success: {name}")

        return pick_point_tf,place_point_tf

    def reward(self):
        pass