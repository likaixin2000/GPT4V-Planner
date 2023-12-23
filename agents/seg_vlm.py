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

from .agent import Agent, PlanResult



class SegVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by the robot's camera, in which some objects are highlighted with masks and marked with numbers. Output your plan as code.

Operation list:
{action_space}

Things to keep in mind:
{additional_meta_prompt}

Note:
- A python list `regions` will be provided for you to reference the objects. Please use the format of `obj=regions[number]`.
- Do not define the operations or regions in your code. They will be provided in the python environment.
- Your code should be surrounded by a python code block "```python".
'''

    def __init__(self, 
        segmentor: Segmentor, 
        vlm: LanguageModel,
        configs: dict = None,
        **kwargs
    ):
        if not isinstance(segmentor, Segmentor):
            raise TypeError("`segmentor` must be an instance of Segmentor.")
        if not isinstance(vlm, LanguageModel):
            raise TypeError("`vlm` must be an instance of LanguageModel.")

        self.segmentor = segmentor
        self.vlm = vlm

        # Default configs
        self.configs = {
            "label_mode": "1",
            "alpha": 0.05
        }
        if configs is not None:
            self.configs = self.configs.update(configs)  

        super().__init__(**kwargs)


    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))

        # Generate segmentation masks
        masks = self.segmentor.segment_auto_mask(image)
        self.log(name="Segment auto mask result", log_type="data", content=masks)
        

        # Draw masks
        # sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        annotated_img = visualize_masks(
            image, 
            annotations=[anno["segmentation"] for anno in masks],
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=False, 
            draw_mark=True, 
            draw_box=False
        )
        self.log(name="Segment auto mask", log_type="call", image=annotated_img)
        
        meta_prompt = self.meta_prompt.format(action_space=self.action_space, additional_meta_prompt = self.additional_meta_prompt)
        self.log(name="VLM call", log_type="call", message=f"Prompt:\n{prompt},\n Meta prompt:\n{meta_prompt}", image=annotated_img)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=meta_prompt
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)
        
        plan_code, filtered_masks = self.extract_plans_and_regions(plan_raw, masks)
        
        return PlanResult(
            success=True,
            plan_code=plan_code,
            masks=filtered_masks,
            plan_raw=plan_raw,
            annotated_image=annotated_img,
            prompt=prompt,
            info_dict=dict(configs=self.configs)
        )