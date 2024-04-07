from utils import logging
from .simulation.environment import Environment
from .simulation.task import Task
from .simulation import task_dic
from agents.agent import Agent
from utils.masks import Mask
from utils.image_utils import visualize_image
import cv2
import numpy as np

def find_mask_center_point(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255  # Make cv2 happy
    height, width = binary_mask.shape
    binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
    mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
    mask_dt = mask_dt[1:-1, 1:-1]
    max_dist = np.max(mask_dt)
    coords_y, coords_x = np.where(mask_dt == max_dist)
    # Only take one
    coords_x = coords_x[0]
    coords_y = coords_y[0]
    
    return coords_x / width, coords_y / height


class ISAACSimulationEnv():
    def __init__(
            self,
            enable_logging: bool = True
        ):
        self.task:Task=None
        # self._isaac_sim_env = None
        self.enable_logging = enable_logging
        self.logger = logging.get_logger() if enable_logging else None

    def setup(self,task_name:str):
        self.task = task_dic[task_name]()
        self.task.reset()

    # return RGB image
    def get_image(self):
        return self.task.get_image()

    def get_execution_context(self,agent:Agent,init_image=None,*args,**kwargs):
        
        grasper_holding = None 
        def pick(obj):
            nonlocal grasper_holding
            if grasper_holding is not None:
                raise Exception("Trying to pick an object when the grasper is currently holding an object.")
            
            mask: Mask = obj
            grasper_holding = mask
    
        
        def place(obj,orientation="on_top_of",*args,**kwargs):
            nonlocal grasper_holding
            if grasper_holding is None:
                raise Exception("Trying to place an object when the grasper is not holding anything.")
            
            # place_mask: Mask = obj
            place_mask = Mask(mask=obj,ref_image=init_image,name='place_mask')
            pick_point = find_mask_center_point(grasper_holding)
            if hasattr(agent, "query_place_position"):
                # Ask VLM to find a good position to place the object
                place_point = agent.query_place_position(
                    # Do not update image here
                    mask=place_mask,
                    intervals=(4, 4), 
                    margins=(0.06, 0.1),
                    orientation=orientation,

                )
            else:
                place_point = find_mask_center_point(obj)

            if init_image is not None:
                visualize_image(image=init_image,points=[place_point])

            self.task.step(pick_point,place_point)
            picked_obj_mask = grasper_holding
            grasper_holding = None
            # todo ???
            # picked_obj_mask.reidentify(new_image=self.get_image(), place_point=place_point, detector=agent.detector)

            if self.logger is not None:
                self.logger.log(name="After action `place`", log_type="action", message=f"Pick point:{pick_point}\nPlace point:{place_point}", image=self.get_image())

        tools = {
            "pick": pick,
            "place": place
        }

        return tools

            
