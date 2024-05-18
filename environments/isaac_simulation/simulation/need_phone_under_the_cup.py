import os
from PIL import Image
from .task import Task
import numpy as np

class NeedPhoneUnderCup(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        # self._env.add_object_relative_to_table("laptop","laptop/laptop.urdf",[-0.1, 0.3, 0.01],[0.09, 0.09, 2, 0.5 * np.pi],fix_base_link=True)

        self._env.add_box_relative_to_table("block",[0.2, 0.2, 0.00],[-0.1, 0.3, 0.01],[0, 0, 1, 0],[0, 0, 0],fix_base_link=True)

        # self._env.add_object_relative_to_table("headphones","earphone/headphones.urdf",[0.1, -0.1, 0.2],[1,0,0,0.5 * np.pi])
        self._env.add_object_relative_to_table("cup","yellow_cup/model.urdf",[-0.1, -0.2, 0.1],[0., 0, 1, 0.5 * np.pi])

        self._env.add_object_relative_to_table("iphone","iphone/iphone.urdf",[-0.1, -0.2, 0.05],[0., 0, 1, -0.5 * np.pi])

        # add distractor
        self._env.add_object_relative_to_table("bowl","bowl/model.urdf",[0.1, -0.1, 0.05],[0., 0, 1, 0.5 * np.pi])
        # self._env.set_look_ahead_camera()
        self._env.set_look_down_degree_camera(45)
        self._env.empty_step(60)
        self._env.set_franka()



    def get_image(self):
        return super().get_image()