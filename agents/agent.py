import json
import re
from typing import List, Optional, Dict, Any

from PIL import Image

from utils.logging import CustomLogger, get_logger
from utils.exceptions import *
from utils.masks import Mask

DEFAULT_ACTION_SPACE = """
 - pick(obj)
 - place(obj, orientation, offset=None). 
   - `obj` is the destination object, around which you want to put down the object the robot is currently holding. 
   - `orientation` in ['inside', 'on_top_of', 'left', 'right', 'up', 'down']
   - `offset`: Normalized positions from 0 to 1 within the object region. [0.0, 0.0] is the top-left corner and [1.0, 1.0] is the bottom-right corner. 
Only applicable when orientation="inside" or "on_top_of". You should decide carefully on this to avoid object collisions.
 - open(obj)
"""

COMMON_PROMPT = """
- You are REQUIRED to pick an object ONLY when there are no other objects stacked on top of it.
- If the operation is not achivable, or the object you want to interact with does not have a number marked on it, state the reason and do not return any code.
"""

class PlanResult:
    def __init__(
        self, 
        success: bool = False, 
        exception: Optional[Exception] = None, 
        plan_raw: Optional[str] = None, 
        masks: Optional[list[Mask]] = None, 
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
                f"masks=<{len(self.masks) if self.masks is not None else 'No'} masks>,\n "
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
            additional_meta_prompt: str = COMMON_PROMPT,
            enable_logging: bool = True,
            ) -> None:
        self.action_space = action_space
        self.additional_meta_prompt = COMMON_PROMPT
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

    def extract_code_block(self, text: str):
        # Extract code blocks. We assume there is only one code block in the generation
        code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        if not code_blocks:
            raise EmptyCodeError("No python code block was found.")

        code_block = code_blocks[0]
        return code_block

    def extract_regions_of_interest(self, code: str):
        matches = re.findall(r'regions\[(\d+)\]', code)

        refs = list(set(int(index) for index in matches))
        refs.sort()
        return refs

    def extract_plans_and_regions(self, text: str, regions: list, *lists):
        """
        Extracts a Python code block from the llm/vlm's response and updates the region index references within the code.

        This method locates a Python code block within the provided text, assuming there is only one such block.
        It then finds all occurrences of region index references in the format 'obj=regions[x]', where 'x'
        is an integer. These indices are normalized to create a continuous sequence starting from 0. The method
        also filters the regions list based on the indices used in the code, with the understanding that the
        indices start from 1 in the code.

        Parameters:
        text (str): The text string containing the Python code block.
        regions (list): The list of regions that are referenced in the code block.
        lists: Other arrays that requires remapping.

        Returns:
        tuple: A tuple containing two elements:
            - The modified code block with updated region index references.
            - A list of filtered regions based on the indices used in the code block.

        Raises:
        EmptyCodeError: If no Python code block is found in the text.
        BadCodeError: If an invalid region index is referenced in the code block.

        Example:
        >>> text = "Some text...```python\npick(obj=regions[3])\n```...more text"
        >>> regions = ['Region1', 'Region2', 'Region3', 'Region4']
        >>> extract_plans_and_regions(text, regions)
        ("pick(obj=regions[2])", ['Region3'])
        """
        self.log(name="Extract plan code and filtered masks", log_type="call")

        code_block = self.extract_code_block(text)
        refs = self.extract_regions_of_interest(code_block)

        # Remap the regions with continuous ascending indices.
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(refs)}
        for old_index, new_index in index_mapping.items():
            code_block = code_block.replace(f'regions[{old_index}]', f'regions[{new_index}]')
        try:
            filtered_regions = [regions[index - 1] for index in refs]  # indices starts from 1 !!!!!
        except IndexError as e:  # Invalid index is used
            raise BadCodeError("Invalid region index is referenced.")

        # Remap additional arrays
        remapped_arrays = []
        for li in lists:
            remapped_arrays.append([li[index - 1] for index in refs])

        self.log(name="Extracted plan code", log_type="info", message=code_block)
        self.log(name="Extracted masks", log_type="data", content=filtered_regions)
        return code_block, filtered_regions, *remapped_arrays
    