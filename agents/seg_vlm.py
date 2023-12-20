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

Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
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
            "img_size": None,
            "label_mode": "1",
            "alpha": 0.05
        }
        if configs is not None:
            self.configs = self.configs.update(configs)  

        super().__init__(**kwargs)


    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))

        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs and self.configs["img_size"]:
            processed_image = resize_image(image, self.configs["img_size"])

        # Generate segmentation masks
        self.log(name="Segment auto mask", log_type="call", image=processed_image)
        masks = self.segmentor.segment_auto_mask(processed_image)
        self.log(name="Segment auto mask result", log_type="data", content=masks)
        

        # Draw masks
        # sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        annotated_img = visualize_masks(
            processed_image, 
            annotations=[anno["segmentation"] for anno in masks],
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=False, 
            draw_mark=True, 
            draw_box=False
        )
        
        meta_prompt = self.meta_prompt.format(action_space=self.action_space)
        self.log(name="VLM call", log_type="call", message=f"Prompt: {prompt},\n Meta prompt: {meta_prompt}", image=annotated_img)
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