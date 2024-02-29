import json
import re
from typing import List, Optional, Dict, Any

from PIL import Image

from apis.language_model import LanguageModel
from apis.detectors import Detector, COMMON_OBJECTS
from apis.segmentors import Segmentor

from utils.image_utils import annotate_masks, get_visualized_image
from utils.logging import CustomLogger, get_logger
from utils.exceptions import *
from utils.masks import Mask

from .agent import Agent, PlanResult


class DetLLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will be given a list of objects detected which you may want to interact with. Output your plan as code.

Operation list:
{action_space}

Things to keep in mind:
{additional_meta_prompt}

Note:
- For any item referenced in your code, please use the format of `obj="object_name"`.
- Do not define the operations in your code. They will be provided in the python environment.
- Your code should be surrounded by a python code block "```python".
'''

    def __init__(
            self, 
            detector: Detector,
            segmentor: Segmentor, 
            llm: LanguageModel,
            configs: dict = None,
            **kwargs
            ):
        if not isinstance(detector, Detector):
            raise TypeError("`detector` must be an instance of Detector.")
        if not isinstance(segmentor, Segmentor):
            raise TypeError("`segmentor` must be an instance of Segmentor.")
        if not isinstance(llm, LanguageModel):
            raise TypeError("`llm` must be an instance of LanguageModel.")

        self.detector = detector
        self.segmentor = segmentor
        self.llm = llm

        # Default configs
        self.configs = {
            "alpha": 0.7,
            "include_coordinates": True
        }
        # Configs
        if configs is not None:
            self.configs = self.configs.update(configs)


        super().__init__(**kwargs)

    def extract_regions_of_interest(self, code: str):
        # In DetLLM, the references are object names.
        matches = re.findall(r'obj=[\'"](.+?)[\'"]', code)

        refs = list(set(object_name for object_name in matches))
        refs.sort()
        return refs

    def textualize_detections(self, detected_objects: list, include_coordinates=False) -> str:
        """
        Creates a Markdown formatted list of detected object names, with an option to include normalized position coordinates.

        Args:
            detected_objects (list of dict): A list of dictionaries, each representing a detected object.
                                            Each dictionary should have a 'box_name' key, and optionally a 'box' key with normalized coordinates (ranging from 0 to 1).
            include_coordinates (bool): If True, includes the positions (normalized coordinates) of the detected objects in the list, if available.

        Returns:
            str: A Markdown formatted string listing the detected object names, optionally with their normalized position coordinates.

        Example:
            Sample input:
                example_detections = [
                    {'box_name': 'Cat', 'box': [0.1, 0.15, 0.2, 0.25]},
                    {'box_name': 'Dog', 'box': [0.3, 0.35, 0.4, 0.45]},
                    {'box_name': 'Bird', 'box': [0.05, 0.075, 0.12, 0.145]},
                    {'box_name': 'Car', 'box': [0.5, 0.55, 0.6, 0.65]}
                ]
                markdown_list = textualize_detections(example_detections, include_coordinates=True)

            Sample output:
                - Cat (coordinates: (0.1, 0.15), (0.2, 0.25))
                - Dog (coordinates: (0.3, 0.35), (0.4, 0.45))
                - Bird (coordinates: (0.05, 0.075), (0.12, 0.145))
                - Car (coordinates: (0.5, 0.55), (0.6, 0.65))
        """

        markdown_list = []
        if include_coordinates:
            markdown_list.append("List of objects detected (coordinates are in (x1,y1), (x2, y2) order):")
        else:
            markdown_list.append("List of objects detected:")

        for obj in detected_objects:
            box_name = obj['box_name']
            if include_coordinates:
                box = obj['bbox']
                box_coords = f" (coordinates: ({box[0]:.2f}, {box[1]:.2f}), ({box[2]:.2f}, {box[3]:.2f}))"
                markdown_list.append(f"- {box_name}{box_coords}")
            else:
                markdown_list.append(f"- {box_name}")

        result = '\n'.join(markdown_list)
        return result

    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))
        
        # Detect boxes
        text_queries = COMMON_OBJECTS
        detected_objects = self.detector.detect_objects(
            image,
            text_queries,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.2
        )

        self.log(name="Detected objects", log_type="data", content=detected_objects)
        # Check if the detections are complete
        #  1. It should not be empty
        if len(detected_objects) == 0:
            raise EmptyObjectOfInterestError("No objects were detected in the image.")
        
        #  2. No multiple instances
        # Convert the detections to a dict and check if there are multiple instances of the same name.
        detected_objects_dict = {}
        for detection in detected_objects:
            object_name = detection["box_name"]
            if object_name not in detected_objects_dict:
                detected_objects_dict[object_name] = detection
            else:
                raise NameConflictError(f"There are multiple instances of object '{object_name}'. DetLLM can not tell them apart.")
        
        # Draw bboxes (for debugging)
        bbox_img = get_visualized_image(
            image,
            bboxes=[obj['bbox'] for obj in detected_objects]
        )
        # Convert detection results to a string
        textualized_object_list = self.textualize_detections(detected_objects, include_coordinates=self.configs["include_coordinates"])
        self.log(name="Detected objects", message=textualized_object_list, image=bbox_img)
        
        # Call LLM
        prompt = textualized_object_list + '\n\n' + prompt
        meta_prompt = self.meta_prompt.format(action_space=self.action_space, additional_meta_prompt = self.additional_meta_prompt)
        self.log(name="LLM call", log_type="call", message=f"Prompt:\n{prompt},\n Meta prompt:\n{meta_prompt}")
        plan_raw = self.llm.chat(
            prompt=prompt, 
            meta_prompt=meta_prompt
        )

        plan_code_raw = self.extract_code_block(plan_raw)
        objects_of_interest = self.extract_regions_of_interest(plan_code_raw)
        self.log(name="Objects of interest", message="\n".join(objects_of_interest))
        # Check if all objects mentioned are in the detections.
        objects_not_detected = []
        for object_name in objects_of_interest:
            if object_name not in detected_objects_dict.keys():
                objects_not_detected.append(object_name)
        if len(objects_not_detected):
            raise MissingObjectError(f"Object(s) of interest ({', '.join(objects_not_detected)}) are not detected.")

        bboxes_of_interest = [detected_objects_dict[name]["bbox"] for name in objects_of_interest]

        # Segment masks
        self.log(name="Segmentor segment_by_bboxes call", log_type="call")
        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=bboxes_of_interest)
        # Visualize and log the result for debugging
        segment_img = annotate_masks(
            image, 
            masks=[anno["segmentation"] for anno in masks],
            label_mode="1",
            alpha=0.8,
            draw_mask=True, 
            draw_mark=True, 
            draw_box=True
        )
        assert len(masks) == len(objects_of_interest)
        self.log(name="Segmentor segment_by_bboxes result", log_type="info", image=segment_img)

        # Convert references with names to `regions[i]`
        plan_code = plan_code_raw[:]  # Make a copy
        for i, object_name in enumerate(objects_of_interest):
            pattern = r'obj=[\'"]' + re.escape(object_name) + r'[\'"]'
            plan_code = re.sub(pattern, f'regions[{i}]', plan_code)

        masks = Mask.from_list(mask_list=masks, names=objects_of_interest, ref_image=image)

        return PlanResult(
            success=True,
            plan_code=plan_code,
            masks=masks,
            annotated_image=segment_img,
            plan_raw=plan_raw,
            prompt=prompt,
            info_dict=dict(configs=self.configs, detected_objects=detected_objects)
        )
