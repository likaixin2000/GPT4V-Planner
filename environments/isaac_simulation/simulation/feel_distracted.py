import os
from PIL import Image
from .task import Task
import numpy as np

class FeelDistracted(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._env.add_object_relative_to_table("laptop","laptop/laptop.urdf",[-0.1, 0.3, 0.01],[0.09, 0.09, 2, 0.5 * np.pi],fix_base_link=True)

        self._env.add_object_relative_to_table("headphones","earphone/headphones.urdf",[0.1, -0.1, 0.2],[1,0,0,0.5 * np.pi])

        self._env.add_object_relative_to_table_eular("earphone_box","earphone/earphone_box.urdf",[0.1, -0.3, 0.2],[0.5 * np.pi,0,-0.5 * np.pi])

        # add distractor
        self._env.add_object_relative_to_table("cup","yellow_cup/model.urdf",[-0.1, -0.2, 0.05],[0., 0, 1, 0.5 * np.pi])
        #绕x轴旋转90度，再绕z轴旋转90度
        # self._env.add_object_relative_to_table("glasses","glasses/glasses.urdf",[0, 0, 0.5],[1,0,0,0.5 * np.pi])
        self._env.add_object_relative_to_table("book","book_1/model.urdf",[0.1, 0.3, 0.2],[0.09, 0.09, 2, 0.5 * np.pi])

        # self._env.set_look_ahead_camera()
        self._env.set_look_down_degree_camera(45)
        self._env.empty_step(60)
        self._env.set_franka()



    def get_image(self):
        return super().get_image()