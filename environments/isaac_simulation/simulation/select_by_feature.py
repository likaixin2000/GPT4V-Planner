import os
from PIL import Image
from .task import Task
import numpy as np
import random

class SelectByFeature(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        # 1. add environments object

        # 2. target object and reward position 
                # self._env.add_object_relative_to_table_eular("glasses","glasses/glasses.urdf",[0, 0, 0.5],[0.5 * np.pi,0,0.5 * np.pi])

        feature_dict = self.object_helper.get_feature_dict()
        self.feature = random.choice(list(feature_dict.keys()))
        help_objects = self.object_helper.get_selects_by_feature(self.feature)
        self.target_object_name,self.target_object_path = random.choice(help_objects)



        self.reference_object = self.object_helper.get_reference_object(unselect_list=help_objects)
        self.place_position_info = self.position_helper.get_random_position()
        

        # 3. define prompt
        self.tast_prompt = f"Put the {self.feature} {self.place_position_info['words']} the {self.reference_object[0]}."
        # 4. add distractor

        unselected_object = [self.reference_object] + help_objects
        self.distactor_objects = self.object_helper.get_distractors(n=2, unselect_list=unselected_object)

        # 5. layout the objects

        self.object_lists = ([(self.target_object_name,self.target_object_path), self.reference_object] + self.distactor_objects)
        random.shuffle(self.object_lists)

        #todo random layout 

        positions = [(-0.15,-0.25),(-0.15,0.25),(0.15,-0.25),(0.15,0.25)]
        
        print(self.object_lists)
        for i, (object_name, object_path) in enumerate(self.object_lists):

            self._env.add_object_relative_to_table_eular(object_name,object_path,[positions[i][0],positions[i][1],0.15],[0,0,0])


        self._env.set_look_down_degree_camera(45)
        self._env.empty_step(60)
        self._env.set_franka()
        print(self.tast_prompt)



    def get_image(self):
        return super().get_image()
    
    def reward(self):
        target_pose = self._env.get_gym_handle_pose(self.target_object_name)
        reference_pose = self._env.get_gym_handle_pose(self.reference_object[0])
        x_y_answer_position = np.array(self.place_position_info["relative_position"]) + np.array(reference_pose[:2])
        x_y_taget_position = np.array(target_pose[:2])
        distance = np.linalg.norm(x_y_answer_position - x_y_taget_position)
        if distance < self.place_position_info["distance"]:
            return 1
        else:
            return 0