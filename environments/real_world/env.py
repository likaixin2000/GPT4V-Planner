from functools import partial

from ...apis.detectors import Detector, OWLViT, COMMON_OBJECTS
from ...apis.effectors import Effector
from environments import Environment, PlanExecutionError

from utils import logging
from utils.image_utils import visualize_image

class RealWorldEnv(Environment):
    def __init__(
            self,
            enable_logging: bool = True
        ):
        self._ur5_sim_env = None

        self.enable_logging = enable_logging
        self.logger = logging.get_logger() if enable_logging else None

    def setup(self, task_name: str):
        pass

    def get_image(self):
        pass

    def get_execution_context(self):
        """Create tools and actions for LLMs to call."""

        # -------------------------------------------------------------------------------------
        # Define tools here.

        # The env accepts pick and place as a single function call.
        # Record the object mask the grasper is currently holding.
        grasper_holding = None

        def pick(obj):
            pass

        def place(obj, orientation='notimplemented'):
            pass
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

        grasper_holding = False

        def pick(obj):
            nonlocal grasper_holding
            if grasper_holding is not None:
                raise PlanExecutionError("Trying to pick an object when the grasper is currently holding an object.")
            grasper_holding = obj

            # Log
            log_image = visualize_image(plan_image, mask=obj)
            inspect_logger.log(name="Action `place`", log_type="action", image=log_image)


        def place(obj, orientation='notimplemented'):
            nonlocal grasper_holding
            if grasper_holding is None:
                raise PlanExecutionError("Trying to place an object when the grasper is currently holding nothing.")
 
            # Log
            log_image = visualize_image(plan_image, mask=obj)
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
 



