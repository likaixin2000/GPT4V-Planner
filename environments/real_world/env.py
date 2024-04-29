from functools import partial

import numpy as np
from PIL import Image

from apis.detectors import Detector, OWLViT, COMMON_OBJECTS
from environments import Environment, PlanExecutionError

from utils import logging
from utils.image_utils import get_visualized_image
from utils.masks import Mask


class RealWorldEnv(Environment):
    def __init__(
            self,
            enable_logging: bool = True
        ):
        self._ur5_sim_env = None

        self.enable_logging = enable_logging
        self.logger = logging.get_logger() if enable_logging else None

        self.effector = None

    def setup(self):
        from service.grasper.grasper import FetchAgent
        self.effector = FetchAgent()

    def get_image(self):
        assert self.effector, "The effector is not initialized. Did you forget to call setup()?"
        image = Image.fromarray(self.effector.get_img())
        return image

    def get_execution_context(self, agent, logger=None):
        """Create tools and actions for LLMs to call."""

        # -------------------------------------------------------------------------------------
        # Define tools here.
        grasper_holding_obj = None

        def pick(obj):
            nonlocal grasper_holding_obj

            mask = obj
            self.effector.pick(mask.mask)
            grasper_holding_obj = mask

        def place(obj, orientation='not implemented', offset='not implemented'):
            nonlocal grasper_holding_obj

            place_obj: Mask = obj
            if hasattr(agent, "query_place_position"):
                # Ask VLM to find a good position to place the object
                place_point = agent.query_place_position(
                    # Do not update image here
                    mask=place_obj,
                    orientation=orientation,
                    num_marks=(3, 3), 
                    margins=(0.1, 0.1)
                )
                if logger:
                    logger.log(name="Query place position", log_type="action", image=get_visualized_image(self.get_image(), masks=[place_obj.mask], points=[place_point]))
            else:
                place_point = obj.find_mask_center_point()

            # Convert place point to integer coordinates
            place_mask = np.zeros_like(place_obj.mask).astype(np.uint8)
            viewport_x, viewport_y = place_mask.shape[1], place_mask.shape[0]
            place_point_int = (int(viewport_x * place_point[0]), int(viewport_y * place_point[1]))
            # Erode the mask around the center point
            place_mask[place_point_int[1]-5:place_point_int[1]+5, place_point_int[0]-5:place_point_int[0]+5] = 1
            self.effector.placeon(place_mask)

            # Re-identify the picked object and update the mask
            grasper_holding_obj.reidentify(new_image=self.get_image(), place_point=place_point, detector=agent.detector, segmentor=agent.segmentor)

            grasper_holding_obj = None

        # End of tools definition
        # -------------------------------------------------------------------------------------

        tools = {
            "pick": pick,
            "place": place
        }

        return tools
    
    def get_inspect_execution_context(self, plan_image):
        """
        This function creates a fake execution context. 
        Executing the plan code in this context generates a sequence of images and text for the user to inspect.

        Parameters:
        - image (PIL.Image): The exact image used for planning. 
        It is not a good idea to get a new image because of the postprocessing it may have undergone.
        """
        # Crete the logger that records all actions
        inspect_logger = logging.CustomLogger("Inspector")

        grasper_holding = None

        def pick(obj):
            nonlocal grasper_holding
            if grasper_holding is not None:
                raise PlanExecutionError("Trying to pick an object when the grasper is currently holding an object.")
            grasper_holding = obj

            # Log
            log_image = get_visualized_image(plan_image, masks=[obj.mask])
            inspect_logger.log(name="Action `pick`", log_type="action", image=log_image)


        def place(obj, orientation='not implemented', offset='not implemented'):
            nonlocal grasper_holding
            if grasper_holding is None:
                raise PlanExecutionError("Trying to place an object when the grasper is currently holding nothing.")
            grasper_holding = None
 
            # Log
            log_image = get_visualized_image(plan_image, masks=[obj.mask])
            inspect_logger.log(name="Action `place`", log_type="action", image=log_image)

        # End of tools definition
        # -------------------------------------------------------------------------------------

        tools = {
            "pick": pick,
            "place": place
        }

        return tools, inspect_logger

# def build_execution_env():
#     """Create tools for LLMs to call."""
#     detector = OWLViT()
#     effector = Effector()

#     # -------------------------------------------------------------------------------------
#     # Define tools here.

#     # Perception
#     detect_all_objects = partial(detector.detect_objects(text_queries=COMMON_OBJECTS))
#     detect_object = detector.detect_objects
#     # Manipulation
#     pick = effector.grasp
#     # End of tools definition
#     # -------------------------------------------------------------------------------------

#     tools = {
#         "detect_all_objects": detect_all_objects, 
#         "detect_object": detect_object, 
#         "pick": pick
#     }

#     return tools
 



