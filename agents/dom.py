import json
import re
from typing import  Optional, Dict, Any
# from typing import List as list

from PIL import Image
import numpy as np

from apis.language_model import LanguageModel
from apis.detectors import Detector, COMMON_OBJECTS
from apis.segmentors import Segmentor

from utils.image_utils import resize_image, annotate_masks, get_visualized_image, annotate_positions_in_image
from utils.logging import CustomLogger, get_logger
from utils.exceptions import *
from utils.masks import Mask

from .agent import Agent, PlanResult


class DOM(Agent):
    meta_prompt_plan = \
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
- For any item referenced in your code, please use the format of `obj="object_name"`.
- Your object list should be encompassed by a json code block "```json".
- Your code should be surrounded by a python code block "```python".
'''
    meta_prompt_inspect = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by the robot's camera, in which some objects are highlighted with masks and marked with numbers.
The plan will be given to you as python code. Your job is to replace the all `object` parameters to the correct region numbers. Then, output your final plan code.

Operation list:
{action_space}

Things to keep in mind:
{additional_meta_prompt}

Note:
- A python list `regions` will be provided for you to reference the objects. Please use the format of `obj=regions[number]`.
- Do not define the operations or regions in your code. They will be provided in the python environment.
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
        self.image = image

        # Generate a response from VLM
        meta_prompt_plan = self.meta_prompt_plan.format(action_space=self.action_space, additional_meta_prompt = self.additional_meta_prompt)
        self.log(name="VLM call", log_type="call", message=f"Prompt:\n{prompt},\n Meta prompt:\n{meta_prompt_plan}", image=image)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=image, 
            meta_prompt=meta_prompt_plan
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

        # NOTE: This requires the object names to be unique.
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
        annotated_img = annotate_masks(
            image, 
            masks=[anno["segmentation"] for anno in masks],
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=False, 
            draw_mark=True, 
            draw_box=False
        )
        self.log(name="Segmentor segment_by_bboxes result", log_type="info", image=annotated_img)

        # Ask the VLM to inspect the masks to disambiguate the objects
        # This object has no state. It will be a new conversation.
        meta_prompt_inspect = self.meta_prompt_inspect.format(action_space=self.action_space, additional_meta_prompt = self.additional_meta_prompt)
        self.log(name="VLM call: final plan", log_type="call", message=f"Prompt:\n{prompt},\n Meta prompt:\n{meta_prompt_inspect}", image=image)
        final_plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=meta_prompt_inspect
        )
        self.log(name="Final raw plan", log_type="info", message=final_plan_raw)

        plan_code, filtered_masks, box_names = self.extract_plans_and_regions(final_plan_raw, masks, objects_of_interest)

        masks = Mask.from_list(mask_list=masks, names=box_names, ref_image=image)

        return PlanResult(
            success=True,
            plan_code=plan_code,
            masks=filtered_masks,
            plan_raw=final_plan_raw,
            annotated_image=annotated_img,
            prompt=prompt,
            info_dict=dict(
                configs=self.configs, 
                plan_raw_before_inspect=plan_raw
            )
        )
    

    def query_place_position(
            self, 
            mask: Mask,
            intervals = (3, 3), 
            margins = (3, 3),
            orientation: str = "on_top_of"
    ):
        # Crop the object placed on
        cropped_image, cropped_box = mask.crop_obj(padding=0.3)

        x_interval, y_interval = intervals
        x_margin, y_margin = margins
        x = np.linspace(0 + x_margin, 1 - x_margin, x_interval)
        y = np.linspace(0 + y_margin, 1 - y_margin, y_interval)

        # Create the meshgrid
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        positions = np.vstack([X, Y]).T
        
        height, width = mask.mask.shape
        viewport_size = (
            width * (cropped_box[2] - cropped_box[0]), 
            height * (cropped_box[3] - cropped_box[1])
        )

        annotated_image = annotate_positions_in_image(cropped_image, positions, font_size=min(viewport_size) * 0.1)
        
        # Query VLM
        position_response = self.vlm.chat(
            prompt=f"Please carefully review the image with numbers marked on it to denote locations. It shows a {mask.name} and I want to put an object (not shown in the image) {orientation} of it. You should select the safest and suitable position for placing it, ensuring there is no collision with existing elements. Choose and output only one number representing the position.", 
            image=annotated_image, 
            meta_prompt=""
        )
        position_index = int(position_response.strip()) - 1
        position = positions[position_index]

        # Restore the location to the original image
        cropped_box_width = cropped_box[2] - cropped_box[0]
        cropped_box_height = cropped_box[3] - cropped_box[1]
        position = [
            cropped_box[0] + position[0] * cropped_box_width, 
            cropped_box[1] + position[1] * cropped_box_height
        ]

        return position
