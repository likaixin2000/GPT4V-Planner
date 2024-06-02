import os
import sys
import numpy as np
import yaml
import random

DEFAULT_PATH=str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yml"))
# print(DEFAULT_PATH)

POSITION_DICT=[
    {
        "words":'in front of',
        "relative_position":np.array([-0.1,0]),
        "distance":0.1,
    },

    {
        "words":'behind',
        "relative_position":np.array([0.1,0]),
        "distance":0.1,
    },

    {
        "words":'on the left of',
        "relative_position":np.array([0,0.1]),
        "distance":0.1,
    },

    {
        "words":'on the right of',
        "relative_position":np.array([0,-0.1]),
        "distance":0.1,
    }
]



class PositionHelper():
    def __init__(self):
        self.position_dict = POSITION_DICT

    def get_random_position(self):
        return random.choice(self.position_dict)

class ObjectHelper():
    def __init__(self,config_path = DEFAULT_PATH):
        # print(config_path)
        with open(config_path,'r') as f:
            self.config_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.feature_dict = self.config_dict["feature_dict"]
        
        for k,v in self.feature_dict.items():
            self.feature_dict[k] = [tuple(i) for i in v]
        
        ## todo: 应该还是改个名字
        self.under_above_dict = self.config_dict["under_above_dict"]
        for k,v in self.under_above_dict.items():
            self.under_above_dict[k] = [tuple(i) for i in v]

        self.object_list = self.config_dict["object_list"]
        self.object_list = [tuple(i) for i in self.object_list]

    def print_object_list(self):
        print(self.object_list)



    def print_feature_dict(self):
        print(self.feature_dict)

    def get_object_list(self):
        return self.object_list

    def get_feature_dict(self):
        return self.feature_dict

    def get_feature_keys(self):
        return self.feature_dict.keys()
    
    def get_above_under_dict(self):
        return self.under_above_dict



    def get_selects_by_feature(self,feature):
        return self.feature_dict[feature]
    
    def get_reference_object(self,unselect_list=[]):
        return random.choice([object for object in self.object_list if object not in unselect_list])
    
    def get_distractors(self, n:int, unselect_list=[]):
        selectable_objects = [object for object in self.object_list if object not in unselect_list]
        return random.sample(selectable_objects, min(n, len(selectable_objects)))



if __name__ == "__main__":
    position_helper = PositionHelper()
    print(position_helper.get_random_position())
    object_helper = ObjectHelper()
    object_helper.print_object_list()
    object_helper.print_feature_dict()
    print(object_helper.get_feature_keys())
    a = object_helper.get_selects_by_feature("object for drink")
    print(a)
    print(object_helper.get_distractors(3, unselect_list=a))

    


    






        