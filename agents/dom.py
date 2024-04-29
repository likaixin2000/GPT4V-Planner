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
You are in charge of controlling a robot to solve a user task. The image from the robot's camera is provided. You should output the plan with the allowed actions, and then the objects used or interacted with in your plan (excluding the robot itself). 

The object list should be a valid json format, for example, [{{"name": "marker"}}, {{"name": "remote"}}, ...].

Allowed actions:
- pick(obj)
- place(target_obj, orientation)

Note:
- Your object list should be encompassed by a json code block "```json".
- Always use the basic name (only the noun) of the objects because the object detector can only work with simple names. No need to disambiguate them. For instance, use "apple" instead of "red_apple". 

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
            "alpha": 0.25,
            "draw_mask": False,
            "draw_mark": True,
            "draw_box": True,
            "mark_position": "top_left"
        }
        if configs is not None:
            self.configs = self.configs.update(configs)

        super().__init__(**kwargs)

    def extract_objects_of_interest_from_vlm_response(self, plan_raw: str):
        self.log(name="Extract object names of interest", log_type="call")
        json_blocks = re.findall(r'```json(.*?)```', plan_raw, re.DOTALL)
        if not json_blocks:
            raise EmptyObjectOfInterestError("No object of interest found.")
        json_block = json_blocks[0]
        object_names_and_aliases = json.loads(json_block)
        if not object_names_and_aliases:
            raise EmptyObjectOfInterestError("No object of interest found.")
        
        self.log(name="Extracted objects of interest", log_type="info", message=repr(object_names_and_aliases))
        return object_names_and_aliases


    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))
        self.image = image

        # Generate a response from VLM
        meta_prompt_plan = self.meta_prompt_plan
        self.log(name="VLM call: objects of interest", log_type="call", message=f"Meta Prompt:\n{meta_prompt_plan}\n\n Prompt:\n{prompt}", image=image)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=image, 
            meta_prompt=meta_prompt_plan
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)
        # Extract objects of interest from VLM's response
        object_names_and_aliases = self.extract_objects_of_interest_from_vlm_response(plan_raw)
        objects_of_interest = list(set([obj["name"] for obj in object_names_and_aliases]))
        # Detect only the objects of interest
        self.log(name="Detect objects", log_type="call", message=f"Queries: {objects_of_interest}")
        detected_objects = self.detector.detect_objects(
            image,
            objects_of_interest,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.1
        )
        self.log(name="Detected objects", log_type="data", content=detected_objects)
        # Draw the masks for logging
        logging_image = get_visualized_image(image, bboxes=[obj["bbox"] for obj in detected_objects])
        self.log(name="Detected objects", log_type="info", image=logging_image, message="\n".join([obj["box_name"] for obj in detected_objects]))

        # NMS to supress overlapping boxes of the same object
        detected_objects = nms(detected_objects)

        # Check if any object of interest is missing in the detected objects
        detected_object_names = [obj["box_name"] for obj in detected_objects]
        missing_objects = set(objects_of_interest) - set(detected_object_names)
        if missing_objects:
            raise MissingObjectError(f"Missing objects that were not detected or had no best box: {', '.join(missing_objects)}")
        
        segment_results = self.segmentor.segment_by_bboxes(image=image, bboxes=[obj["bbox"] for obj in detected_objects])
        masks=[anno["segmentation"] for anno in segment_results]
        annotated_img = annotate_masks(
            image, 
            masks=masks,
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=self.configs["draw_mask"], 
            draw_mark=self.configs["draw_mark"], 
            draw_box=self.configs["draw_box"],
            mark_position=self.configs["mark_position"]
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

        plan_code, filtered_masks, box_names = self.extract_plans_and_regions(final_plan_raw, masks, detected_object_names)

        filtered_masks = Mask.from_list(mask_list=filtered_masks, names=box_names, ref_image=image)

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
    
    def query_place_position(
            self, 
            mask: Mask,
            num_marks = (3, 3), 
            margins = (0.1, 0.1),
            orientation: str = "on_top_of"
    ):
        # Crop the object placed on
        cropped_image, cropped_box = mask.crop_obj(padding=0.5)

        x_marks, y_marks = num_marks
        x_margin, y_margin = margins
        x = np.linspace(0 + x_margin, 1 - x_margin, x_marks)
        y = np.linspace(0 + y_margin, 1 - y_margin, y_marks)

        # Create the meshgrid
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        positions = np.vstack([X, Y]).T
        
        height, width = mask.mask.shape
        viewport_size = (
            int(width * (cropped_box[2] - cropped_box[0])), 
            int(height * (cropped_box[3] - cropped_box[1]))
        )
        annotated_image = annotate_positions_in_image(cropped_image, positions, font_size=min(viewport_size) * 0.1)
        self.log(name="VLM call: query place point", log_type="call", image=annotated_image)
        # Query VLM
        position_response = self.vlm.chat(
            prompt=f"Please carefully review the image with numbers marked on it to denote locations. It shows a {mask.name} and I want to put an object (not shown in the image) with the orientation: '{orientation}'. You should select the safest and suitable position for placing it, ensuring there is no collision with existing elements. Choose and output only one number representing the position.", 
            image=annotated_image, 
            meta_prompt="",
        )
        position_index = int(position_response.strip()) - 1
        position_in_cropped_image = positions[position_index]

        # Restore the location to the original image
        cropped_box_width = cropped_box[2] - cropped_box[0]
        cropped_box_height = cropped_box[3] - cropped_box[1]
        position = [
            cropped_box[0] + position_in_cropped_image[0] * cropped_box_width, 
            cropped_box[1] + position_in_cropped_image[1] * cropped_box_height
        ]

        self.log(name="Query place point result", log_type="result", message=str(position_response), image=get_visualized_image(annotated_image, points=[position_in_cropped_image]))

        return position


def compute_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def nms(detected_objects, iou_threshold=0.2):
    # Filter detections by their confidence scores
    detected_objects = sorted(detected_objects, key=lambda x: x['score'], reverse=True)
    
    # List to store the best bounding boxes after NMS
    best_boxes = []

    # Iterate through all the detected objects
    while detected_objects:
        # Pick the detection with the highest score and remove it from the list
        best_box = detected_objects.pop(0)
        best_boxes.append(best_box)

        # Filter out detections that overlap too much with the best box and are of the same object type
        detected_objects = [
            box for box in detected_objects
            if box['box_name'] != best_box['box_name'] or compute_iou(best_box['bbox'], box['bbox']) < iou_threshold
        ]

    return best_boxes