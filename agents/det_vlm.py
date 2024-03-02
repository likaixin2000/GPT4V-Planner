import json
import re
from typing import List, Optional, Dict, Any

from PIL import Image

from apis.language_model import LanguageModel
from apis.detectors import Detector, COMMON_OBJECTS
from apis.segmentors import Segmentor

from utils.image_utils import resize_image, annotate_masks
from utils.logging import CustomLogger, get_logger
from utils.exceptions import *
from utils.masks import Mask

from .agent import Agent, PlanResult


class DetVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by the robot's camera, in which some objects are highlighted with bounding boxes and marked with numbers. Output your plan as code.

Operation list:
{action_space}

Things to keep in mind:
{additional_meta_prompt}

Note:
- A python list `regions` will be provided for you to reference the objects. Please use the format of `obj=regions[number]`.
- Do not define the operations or regions in your code. They will be provided in the python environment.
- Your code should be surrounded by a python code block "```python".
'''

    def __init__(
            self, 
            detector: Detector, 
            segmentor: Segmentor,
            vlm: LanguageModel,
            configs: dict = None,
            **kwargs
            ):
        if not isinstance(detector, Detector):
            raise TypeError("`detector` must be an instance of Detector.")
        if not isinstance(segmentor, Segmentor):
            raise TypeError("`segmentor` must be an instance of Segmentor.")
        if not isinstance(vlm, LanguageModel):
            raise TypeError("`vlm` must be an instance of LanguageModel.")

        self.detector = detector
        self.segmentor = segmentor
        self.vlm = vlm

        # Default configs
        self.configs = {
            "label_mode": "1",
            "alpha": 0.75
        }
        if configs is not None:
            self.configs = self.configs.update(configs)            

        super().__init__(**kwargs)
    
    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))

        # Generate detection boxes
        text_queries = COMMON_OBJECTS
        self.log(name="Detect objects", log_type="call", message=f"Queries: {text_queries}", image=image)
        detected_objects = self.detector.detect_objects(
            image,
            text_queries,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.2
        )

        self.log(name="Detected objects", log_type="data", content=detected_objects)
        
        if len(detected_objects) == 0:
            raise EmptyObjectOfInterestError("No objects were detected in the image.")

        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[obj['bbox'] for obj in detected_objects])
        # Draw masks
        annotated_img = annotate_masks(
            image,
            masks=[mask['segmentation'] for mask in masks], 
            alpha=self.configs["alpha"],
            draw_box=True,
            draw_mask=False
        )
        
        meta_prompt = self.meta_prompt.format(action_space=self.action_space, additional_meta_prompt = self.additional_meta_prompt)
        self.log(name="VLM call", log_type="call", message=f"Prompt:\n{prompt},\n Meta prompt:\n{meta_prompt}", image=annotated_img)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=meta_prompt
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)

        box_names = [box_data["box_name"] for box_data in detected_objects]
        plan_code, filtered_masks, filtered_names = self.extract_plans_and_regions(plan_raw, masks, box_names)

        masks = Mask.from_list(mask_list=masks, names=box_names, ref_image=image)
        
        return PlanResult(
            success=True,
            plan_code=plan_code,
            masks=filtered_masks,
            plan_raw=plan_raw,
            annotated_image=annotated_img,
            prompt=prompt,
            info_dict=dict(configs=self.configs)
        )
