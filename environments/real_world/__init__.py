# Define the primitives of this environment as Python code context.


from functools import partial
from ...api.detectors import Detector, OWLViT, COMMON_OBJECTS
from ...api.effectors import Effector

def build_execution_env():
    """Create tools for LLMs to call."""
    detector = OWLViT()
    effector = Effector()

    # Define tools here.
    # -------------------------------------------------------------------------------------
    # Perception
    detect_all_objects = partial(detector.detect_objects(text_queries=COMMON_OBJECTS))
    detect_object = detector.detect_objects
    # Manipulation
    pick = effector.grasp
    # End of tools definition
    # -------------------------------------------------------------------------------------

    tools = {
        "detect_all_objects": detect_all_objects, 
        "detect_object": detect_object, 
        "pick": pick
    }

    return tools
 