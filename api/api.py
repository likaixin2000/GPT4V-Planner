from functools import partial
from .detectors import Detector, OWLViT, COMMON_OBJECTS
from .effectors import Effector

def build_execution_env():
    """Create tools for LLMs to call."""
    detector = OWLViT()
    effector = Effector()

    def build_tools():
        """Define tools here."""
        # Perception
        if detector:
            detect_all_objects = partial(detector.detect_objects(text_queries=COMMON_OBJECTS))
            detect_object = detector.detect_objects
        # Manipulation
        pick = effector.grasp

    tools = [
        "detect_all_objects", 
        "detect_object", 
        "pick"
    ]

    env = {k: v for k, v in locals() if k in tools}
    return env
 