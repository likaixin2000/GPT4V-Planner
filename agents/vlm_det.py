import json
import re
from typing import List, Optional, Dict, Any

from PIL import Image

from apis.language_model import LanguageModel
from apis.detectors import Detector, COMMON_OBJECTS
from apis.segmentors import Segmentor

from utils.image_utils import resize_image, annotate_masks, visualize_image, get_visualized_image
from utils.logging import CustomLogger, get_logger
from utils.exceptions import *

from .agent import Agent, PlanResult


class VLMDet(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. 
You need to output your plan as python code.
After writing the code, you should also tell me the objects you want to interact with in your code. To reduce ambiguity, you should try to use different but simple and common names to refer to a single object. 
The object list should be a valid json format, for example, [{{"name": "marker", "aliases": ["pen", "pencil"]}}, {{"name": "remote", "aliases": ["remote controller", "controller"]}}, ...]. "aliases" should be an empty list if there are no aliases.

Operation list:
{action_space}

Things to keep in mind:
{additional_meta_prompt}

Note:
- Do not redefine functions in the operation list.
- A python list `regions` will be provided for you to reference the objects. Please use the format of `obj="name"`.
- Your object list should be encompassed by a json code block "```json".
- Your code should be surrounded by a python code block "```python".
'''

    def __init__(self, vlm: LanguageModel, detector: Detector, segmentor: Segmentor, configs: dict = None, **kwargs):
        if not isinstance(vlm, LanguageModel):
            raise TypeError("`vlm` must be an instance of LanguageModel.")
        if not isinstance(detector, Detector):
            raise TypeError("`detector` must be an instance of Detector.")
        if not isinstance(segmentor, Segmentor):
            raise TypeError("`segmentor` must be an instance of Segmentor.")

        self.vlm = vlm
        self.detector = detector
        self.segmentor = segmentor

        # Default configs
        self.configs = {
            "label_mode": "1",
            "alpha": 0.75
        }
        if configs is not None:
            self.configs = self.configs.update(configs)

        super().__init__(**kwargs)

    def extract_objects_of_interest_from_vlm_response(self, plan_raw: str):
        self.log(name="Extract plan code and object names", log_type="call")
        # Extract code blocks. We assume there is only one code block in the generation
        code_block = self.extract_code_block(plan_raw)
        json_blocks = re.findall(r'```json(.*?)```', plan_raw, re.DOTALL)
        if not json_blocks:
            raise EmptyObjectOfInterestError("No object of interest found.")
        json_block = json_blocks[0]
        object_names_and_aliases = json.loads(json_block)
        if not object_names_and_aliases:
            raise EmptyObjectOfInterestError("No object of interest found.")
        
        self.log(name="Extracted plan code", log_type="info", message=code_block)
        self.log(name="Extracted objects of interest", log_type="info", message=repr(object_names_and_aliases))
        return code_block, object_names_and_aliases

    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))

        # Generate a response from VLM
        meta_prompt = self.meta_prompt.format(action_space=self.action_space, additional_meta_prompt = self.additional_meta_prompt)
        self.log(name="VLM call", log_type="call", message=f"Prompt:\n{prompt},\n Meta prompt:\n{meta_prompt}", image=image)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=image, 
            meta_prompt=meta_prompt
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)

        # Extract objects of interest from VLM's response
        plan_code, object_names_and_aliases = self.extract_objects_of_interest_from_vlm_response(plan_raw)
        objects_of_interest = [obj["name"] for obj in object_names_and_aliases]
        # Detect only the objects of interest
        self.log(name="Detect objects", log_type="call", message=f"Queries: {objects_of_interest}", image=image)
        detected_objects = self.detector.detect_objects(
            image,
            objects_of_interest,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.3
        )
        self.log(name="Detected objects", log_type="data", content=detected_objects)
        # Draw the masks for logging
        logging_image = get_visualized_image(image, bboxes=[obj["bbox"] for obj in detected_objects])
        self.log(name="Detected objects", log_type="info", image=logging_image, message="\n".join([obj["box_name"] for obj in detected_objects]))


        # (kaixin) NOTE: This requires the object names to be unique.
        # Filter and select boxes with the correct name and highest score per name
        best_boxes = {}
        for det in detected_objects:
            box_name = det["box_name"]
            if box_name not in best_boxes or det["score"] > best_boxes[box_name]["score"]:
                best_boxes[box_name] = det
                
        # Check if any object of interest is missing in the detected objects
        missing_objects = set(objects_of_interest) - set(best_boxes.keys())
        if missing_objects:
            raise MissingObjectError(f"Missing objects that were not detected or had no best box: {', '.join(missing_objects)}")
            
        # Arrange boxes in the order of objects_of_interest
        boxes_of_interest = [best_boxes[name] for name in objects_of_interest]

        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[obj["bbox"] for obj in boxes_of_interest])
        segment_img = annotate_masks(
            image, 
            masks=[anno["segmentation"] for anno in masks],
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=True, 
            draw_mark=True, 
            draw_box=True
        )
        self.log(name="Segmentor segment_by_bboxes result", log_type="info", image=segment_img)

        # Replace object names with region masks
        for index, object_name in enumerate(objects_of_interest):
            pattern = re.compile(rf'["\']{re.escape(object_name)}["\']')
            plan_code = pattern.sub(f"regions[{str(index)}]", plan_code)

        return PlanResult(
            success=True,
            plan_code=plan_code,
            masks=masks,
            plan_raw=plan_raw,
            annotated_image=segment_img,
            prompt=prompt,
            info_dict=dict(configs=self.configs)
        )
