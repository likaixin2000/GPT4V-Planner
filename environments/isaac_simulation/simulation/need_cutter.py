import os
from PIL import Image
from .task import Task
import numpy as np

class NeedCutter(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._env.add_object_relative_to_table("gift_box","gift_box/gift_box.urdf",[-0.1, 0.28, 0.2],[1,0,0,0.5 * np.pi])

        self._env.add_object_relative_to_table("cutter","scissors/model.urdf",[0.1, -0.2, 0.2],[0,0,1,0])

        # add distractor
        self._env.add_object_relative_to_table("banana","plastic_banana/model.urdf",[-0.1, -0.2, 0.05],[0., 0, 1, 0.5 * np.pi])
        #绕x轴旋转90度，再绕z轴旋转90度
        # self._env.add_object_relative_to_table("glasses","glasses/glasses.urdf",[0, 0, 0.5],[1,0,0,0.5 * np.pi])
        self._env.add_object_relative_to_table("control","remote_controller/control.urdf",[0.1, 0.3, 0.2],[0.09, 0.09, 2, 0.5 * np.pi])

        # self._env.add_box_relative_to_table("box",[0.05, 0.05, 0.05],[0, -0.1, 0],[0, 0, 1, 0],[0, 0.7, 0.7])
        # self._env.add_box_relative_to_table("box",[0.05, 0.05, 0.05],[0, -0.1, 0],[0, 0, 1, 0],[0, 0.7, 0.7])
  
        # self._env.set_look_ahead_camera()
        self._env.set_look_down_degree_camera(45)
        self._env.empty_step(60)
        self._env.set_franka()



    def get_image(self):
        return super().get_image()