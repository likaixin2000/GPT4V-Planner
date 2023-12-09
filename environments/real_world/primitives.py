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
    if detector:
        detect_all_objects = partial(detector.detect_objects(text_queries=COMMON_OBJECTS))
        detect_object = detector.detect_objects
    # Manipulation
    pick = effector.grasp
    # End of tools definition
    # -------------------------------------------------------------------------------------

    tools = [
        "detect_all_objects", 
        "detect_object", 
        "pick"
    ]

    env = {k: v for k, v in locals() if k in tools}
    return env
 