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


class DetVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by the robot's camera, in which some objects are highlighted with bounding boxes and marked with numbers. Output your plan as code.

Operation list:
{action_space}


Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
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
            "img_size": None,
            "label_mode": "1",
            "alpha": 0.75
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
        
        # Generate detection boxes
        text_queries = COMMON_OBJECTS
        self.log(name="Detect objects", log_type="call", message=f"Queries: {text_queries}", image=processed_image)
        detected_objects = self.detector.detect_objects(
            processed_image,
            text_queries,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.3
        )
        #  Example result:
        # [{'score': 0.3141017258167267,
        # 'bbox': [0.212062269449234,
        # 0.3956533372402191,
        # 0.29010745882987976,
        # 0.08735490590333939],
        # 'box_name': 'roof',
        # 'objectness': 0.09425540268421173}, ...
        # ]
        self.log(name="Detected objects", log_type="data", content=detected_objects)
        
        if len(detected_objects) == 0:
            raise EmptyObjectOfInterestError("No objects were detected in the image.")


        # Draw masks
        annotated_img = visualize_bboxes(
            processed_image,
            bboxes=[obj['bbox'] for obj in detected_objects], 
            alpha=self.configs["alpha"]
        )
        
        meta_prompt = self.meta_prompt.format(action_space=self.action_space)
        self.log(name="VLM call", log_type="call", message=f"Prompt: {prompt},\n Meta prompt: {meta_prompt}", image=annotated_img)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=meta_prompt
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)

        self.log(name="Segmentor segment_by_bboxes call", log_type="call", )
        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[[obj['bbox']] for obj in detected_objects])
        # Visualize and log the result for debugging
        segment_img = visualize_masks(
            processed_image, 
            annotations=[anno["segmentation"] for anno in masks],
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=True, 
            draw_mark=True, 
            draw_box=True
        )
        self.log(name="Segmentor segment_by_bboxes result", log_type="info", image=segment_img)

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
