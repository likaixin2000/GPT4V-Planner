import os
from PIL import Image
from .task import Task
import numpy as np

class NeedCup(Task):
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        # self.handle_map['box']=box_handle
        self._env.add_object_relative_to_table("laptop","laptop/laptop.urdf",[-0.05, 0.3, 0.02],[0.09, 0.09, 2, 0.5 * np.pi],fix_base_link=True)
        self._env.add_object_relative_to_table("cup","yellow_cup/model.urdf",[0.1, -0.2, 0.04],[0., 0, 1, 0.5 * np.pi],fix_base_link=True)
        self._env.add_box_relative_to_table("box",[0.05, 0.05, 0.05],[0, -0.1, 0],[0, 0, 1, 0],[0, 0.7, 0.7])
        self._env.set_look_down_camera()
        self._env.empty_step(60)


    def get_image(self):
        return super().get_image()