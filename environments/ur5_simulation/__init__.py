import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation"))
import time

import numpy as np
from PIL import Image
import cv2

from environments import Environment, PlanExecutionError

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


class UR5SimulationEnv():
    def __init__(self, logger=None):
        self._ur5_sim_env = None
        self.logger = logger

    def setup(self):
        from .simulation.test_simulation import setup_test_simulation
        # This will launch a visualization window
        self._ur5_sim_env = setup_test_simulation(display=True)
        time.sleep(1)  # Wair for things to stablize

    def get_image(self):
        color, depth, segment = self._ur5_sim_env.render(self._ur5_sim_env.camera_config_up)
        # We don't use segment since it is cheating
        image = Image.fromarray(color)
        return image

    def get_execution_context(self):
        """Create tools and actions for LLMs to call."""

        # Define tools here.
        # -------------------------------------------------------------------------------------
        # The env accepts pick and place as a single function call.
        # Record the object mask the grasper is currently holding.
        grasper_holding = None

        def pick(obj):
            nonlocal grasper_holding
            if grasper_holding is not None:
                raise PlanExecutionError("Trying to pick an object when the grasper is currently holding an object.")
            grasper_holding = obj

        def place(obj, orientation='notimplemented'):
            nonlocal grasper_holding
            if grasper_holding is None:
                raise PlanExecutionError("Trying to place an object when the grasper is currently holding nothing.")
            # Translate mask to center point
            pick_point = find_mask_center_point(grasper_holding)
            place_point = find_mask_center_point(obj)
            # Actual call to simulated env
            self._ur5_sim_env.step(pick_point, place_point)
            grasper_holding = None

            # Logging
            if self.logger is not None:
                self.logger.log(name="After action `place`", log_type="action", message=f"Pick point:{pick_point}\nPlace point:{place_point}", image=self.get_image())

        # End of tools definition
        # -------------------------------------------------------------------------------------

        tools = {
            "pick": pick,
            "place": place
        }

        return tools