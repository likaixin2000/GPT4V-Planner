import json
import re
from typing import List, Optional, Dict, Any

from PIL import Image

from apis.language_model import LanguageModel
from apis.detectors import Detector, COMMON_OBJECTS
from apis.segmentors import Segmentor

from utils.image_utils import resize_image, visualize_bboxes, visualize_masks
from utils.logging import CustomLogger, get_logger
from utils.exceptions import *


DEFAULT_ACTION_SPACE = """
 - pick(obj)
 - place(obj, orientation). 
   - `orientation` in ['inside', 'on_top_of', 'left', 'right', 'up', 'down']
 - open(obj)
"""


class PlanResult:
    def __init__(
        self, 
        success: bool = False, 
        exception: Optional[Exception] = None, 
        plan_raw: Optional[str] = None, 
        masks: Optional[list[Any]] = None, 
        prompt: Optional[str] = None, 
        plan_code: Optional[str] = None, 
        annotated_image: Optional[Image.Image] = None, 
        info_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        self.success = success
        self.exception = exception
        self.plan_raw = plan_raw
        self.masks = masks
        self.prompt = prompt
        self.plan_code = plan_code
        self.annotated_image = annotated_image
        self.info_dict = info_dict if info_dict is not None else {}

    def __repr__(self) -> str:
        return ("PlanResult("
                f"success={self.success},\n "
                f"exception={repr(self.exception)},\n"
                f"plan_raw={repr(self.plan_raw)},\n "
                f"masks={self.masks},\n "
                f"prompt={repr(self.prompt)},\n "
                f"plan_code={repr(self.plan_code)},\n "
                f"annotated_image={self.annotated_image},\n "
                f"info_dict={repr(self.info_dict)}"
                ")"
        )


class Agent():
    def __init__(
            self, 
            action_space: str = DEFAULT_ACTION_SPACE,
            enable_logging: bool = True,
            ) -> None:
        self.action_space = action_space
        self.configs = {} if not hasattr(self, "configs") else self.configs
        self.enable_logging = enable_logging
        self.logger = get_logger() if enable_logging else None


    def log(self, *args, **kwargs):
        if self.enable_logging:
            self.logger.log(*args, **kwargs)


    def try_plan(self, *args, **kwargs):
        try:
            plan_result = self.plan(*args, **kwargs)
            self.log(name="Plan result", message=repr(plan_result), log_type="info")
            return plan_result
        # Exceptions related to the planning process. It is not raised in the api calls so we should log it explicitly. 
        except PlanException as exception:
            self.log(name="Plan exception", message=f"{repr(exception)}", log_type="plan_exception", content={"exception": exception})
            return PlanResult(
                success=False,
                exception=exception,
                info_dict={"logs": self.logger.get_logs()} if self.logger is not None else {}
            )
        
        # Other exceptions, such as network errors and api key errors are not caught here.
        # The user should fix the components and then do the planning.


    def extract_plans_and_regions(self, text: str, regions: list):
        self.log(name="Extract plan code and filtered masks", log_type="call")

        # Extract code blocks. We assume there is only one code block in the generation
        code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        if not code_blocks:
            raise EmptyCodeError("No python code block was found.")

        code_block = code_blocks[0]

        # Use regular expression to find all occurrences of region[index]
        matches = re.findall(r'regions\[(\d+)\]', code_block)

        used_indices = list(set(int(index) for index in matches))
        used_indices.sort()

        index_mapping = {old_index: new_index for new_index, old_index in enumerate(used_indices)}
        for old_index, new_index in index_mapping.items():
            code_block = code_block.replace(f'regions[{old_index}]', f'regions[{new_index}]')
        try:
            filtered_regions = [regions[index - 1] for index in used_indices]  # indices starts from 1 !!!!!
        except IndexError as e:  # Invalid index is used
            raise BadCodeError("Invalid region index is referenced.")

        self.log(name="Extracted plan code", log_type="info", message=code_block)
        self.log(name="Extracted masks", log_type="data", content=filtered_regions)
        return code_block, filtered_regions
    