import json
import re
from typing import List, Optional, Dict, Any

from PIL import Image

from apis.language_model import LanguageModel
from apis.detectors import Detector, COMMON_OBJECTS
from apis.segmentors import Segmentor

from utils.image_utils import resize_image, visualize_bboxes, visualize_masks
from utils.logging import CustomLogger
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
            logger: CustomLogger = None
            ) -> None:
        self.action_space = action_space
        self.configs = {} if not hasattr(self, "configs") else self.configs
        self.logger = logger if logger is not None else None


    def log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.log(*args, **kwargs)


    def try_plan(self, *args, **kwargs):
        try:
            return self.plan(*args, **kwargs)
        # Exceptions related to the planning process. It is not raised in the api calls so we should log it explicitly. 
        except PlanException as exception:
            self.log(name="Plan Exception", message=f"{repr(exception)}", log_type="plan_exception", content={"exception": exception})
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


class DetLLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will be given a list of objects detected which you may want to interact with. Output your plan as code.

Operation list:
{action_space}

Note:
- For any item referenced in your code, please use the format of `object="object_name"`.
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
            "img_size": None,
            "alpha": 0.7,
            "include_coordinates": True
        }
        # Configs
        if configs is not None:
            self.configs = self.configs.update(configs)


        super().__init__(**kwargs)

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

        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs and self.configs["img_size"]:
            processed_image = resize_image(image, self.configs["img_size"])
        
        # Generate detection boxes
        text_queries = COMMON_OBJECTS
        detected_objects = self.detector.detect_objects(
            processed_image,
            text_queries,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.5
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
        
        # Draw bboxes (for debugging)
        annotated_img = visualize_bboxes(
            processed_image,
            bboxes=[obj['bbox'] for obj in detected_objects], 
            alpha=self.configs["alpha"]
        )

        # Covert detection results to a string
        textualized_object_list = self.textualize_detections(detected_objects, include_coordinates=self.configs["include_coordinates"])
        self.log(name="Textualized detections", log_type="info", message=textualized_object_list)
        

        prompt = textualized_object_list + '\n\n' + prompt
        meta_prompt = self.meta_prompt.format(action_space=self.action_space)
        self.log(name="LLM call", log_type="call", message=f"Prompt: {prompt},\n Meta prompt: {meta_prompt}")
        plan_raw = self.llm.chat(
            prompt=prompt, 
            meta_prompt=meta_prompt
        )

        self.log(name="Segmentor segment_by_bboxes call", log_type="call")
        masks = self.segmentor.segment_by_bboxes(image=processed_image, bboxes=[[obj['bbox']] for obj in detected_objects])
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
            annotated_image=annotated_img,
            plan_raw=plan_raw,
            prompt=prompt,
            info_dict=dict(configs=self.configs, detected_objects=detected_objects)
        )


class VLMDet(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. 
You need to output your plan as python code.
After writing the code, you should also tell me the objects you want to interact with in your code. To reduce ambiguity, you should try to use different but simple and common names to refer to a single object. 
The object list should be a valid json format, for example, [{{"name": "marker", "aliases": ["pen", "pencil"]}}, {{"name": "remote", "aliases": ["remote controller", "controller"]}}, ...]. "aliases" should be an empty list if there are no aliases.

Operation list:
{action_space}

Note:
- Do not redefine functions in the operation list.
- For any item referenced in your code, please use the format of `object="object_name"`.
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
            "img_size": None,
            "label_mode": "1",
            "alpha": 0.75
        }
        if configs is not None:
            self.configs = self.configs.update(configs)

        super().__init__(**kwargs)

    def extract_objects_of_interest_from_vlm_response(self, plan_raw: str):
        self.log(name="Extract plan code and object names", log_type="call")
        # Extract code blocks. We assume there is only one code block in the generation
        code_blocks = re.findall(r'```python(.*?)```', plan_raw, re.DOTALL)
        json_blocks = re.findall(r'```json(.*?)```', plan_raw, re.DOTALL)
        if not code_blocks:
            raise EmptyCodeError("No python code block found.")
        if not json_blocks:
            raise EmptyObjectOfInterestError("No object of interest found.")
        
        code_block = code_blocks[0]
        json_block = json_blocks[0]
        object_names_and_aliases = json.loads(json_block)
        if not object_names_and_aliases:
            raise EmptyObjectOfInterestError("No object of interest found.")
        
        self.log(name="Extracted plan code", log_type="info", message=code_block)
        self.log(name="Extracted objects of interest", log_type="info", message=repr(object_names_and_aliases))
        return code_block, object_names_and_aliases

    def plan(self, prompt: str, image: Image.Image):
        self.logger.log(name="Configs", log_type="info", message=repr(self.configs))

        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs and self.configs["img_size"]:
            processed_image = resize_image(image, self.configs["img_size"])

        # Generate a response from VLM
        meta_prompt = self.meta_prompt.format(action_space=self.action_space)
        self.log(name="VLM call", log_type="call", message=f"Prompt: {prompt},\n Meta prompt: {meta_prompt}", image=processed_image)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=processed_image, 
            meta_prompt=meta_prompt
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)

        # Extract objects of interest from VLM's response
        plan_code, object_names_and_aliases = self.extract_objects_of_interest_from_vlm_response(plan_raw)
        objects_of_interest = [obj["name"] for obj in object_names_and_aliases]
        # Detect only the objects of interest
        self.log(name="Detect objects", log_type="call", message=f"Queries: {objects_of_interest}", image=processed_image)
        detected_objects = self.detector.detect_objects(
            processed_image,
            objects_of_interest,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.3
        )
        self.log(name="Detected objects", log_type="data", content=detected_objects)
        self.log(name="Detected objects", log_type="info", message=[obj["box_name"] for obj in detected_objects])


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
            raise ObjectNotDetectedError(f"Missing objects that were not detected or had no best box: {', '.join(missing_objects)}")
            
        # Arrange boxes in the order of objects_of_interest
        boxes_of_interest = [best_boxes[name] for name in objects_of_interest]

        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[[bbox] for bbox in boxes_of_interest])
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

        # Replace object names with region masks
        for index, object_name in enumerate(objects_of_interest):
            plan_code = plan_code.replace(object_name, f"regions[{str(index)}]")

        return PlanResult(
            success=True,
            plan_code=plan_code,
            masks=masks,
            plan_raw=plan_raw,
            annotated_image=segment_img,
            prompt=prompt,
            info_dict=dict(configs=self.configs)
        )


class VLMDetInspect(VLMDet):
    meta_prompt_plan = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. 
You need to output your plan as python code.
After writing the code, you should also tell me the objects you want to interact with in your code. To reduce ambiguity, you should try to use different but simple and common names to refer to a single object. 
The object list should be a valid json format, for example, [{{"name": "marker", "aliases": ["pen", "pencil"]}}, {{"name": "remote", "aliases": ["remote controller", "controller"]}}, ...]. "aliases" should be an empty list if there are no aliases.

Operation list:
{action_space}

Note:
- Do not redefine functions in the operation list.
- For any item referenced in your code, please use the format of `object="object_name"`.
- Your object list should be encompassed by a json code block "```json".
- Your code should be surrounded by a python code block "```python".
'''
    meta_prompt_inspect = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by the robot's camera, in which some objects are highlighted with masks and marked with numbers.
The plan will be given to you as python code. Your job is to replace the all `object` parameters to the correct region numbers. Then, output your final plan code.

Operation list:
{action_space}

Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
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
        if "img_size" in self.configs:
            processed_image = resize_image(image, self.configs["img_size"])

        # Generate a response from VLM
        meta_prompt = self.meta_prompt.format(action_space=self.action_space)
        self.log(name="VLM call", log_type="call", message=f"Prompt: {prompt},\n Meta prompt: {meta_prompt}", image=processed_image)
        plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=processed_image, 
            meta_prompt=meta_prompt
        )
        self.log(name="Raw plan", log_type="info", message=plan_raw)
        # Extract objects of interest from VLM's response
        plan_code, object_names_and_aliases = self.extract_objects_of_interest_from_vlm_response(plan_raw)
        objects_of_interest = [obj["name"] for obj in object_names_and_aliases]
        # Detect only the objects of interest
        self.log(name="Detect objects", log_type="call", message=f"Queries: {objects_of_interest}", image=processed_image)
        detected_objects = self.detector.detect_objects(
            processed_image,
            objects_of_interest,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.3
        )
        self.log(name="Detected objects", log_type="data", content=detected_objects)
        self.log(name="Detected objects", log_type="info", message=[obj["box_name"] for obj in detected_objects])


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
            raise ObjectNotDetectedError(f"Missing objects that were not detected or had no best box: {', '.join(missing_objects)}")
            
        # Arrange boxes in the order of objects_of_interest
        boxes_of_interest = [best_boxes[name] for name in objects_of_interest]
        
        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[[bbox] for bbox in boxes_of_interest])
        annotated_img = visualize_masks(
            processed_image, 
            annotations=[anno["segmentation"] for anno in masks],
            label_mode=self.configs["label_mode"],
            alpha=self.configs["alpha"],
            draw_mask=False, 
            draw_mark=True, 
            draw_box=True
        )
        self.log(name="Segmentor segment_by_bboxes result", log_type="info", image=annotated_img)

        # Ask the VLM to inspect the masks to disambiguate the objects
        # This object has no state. It will be a new conversation.
        self.log(name="VLM call: final plan", log_type="call", message=f"Prompt: {prompt},\n Meta prompt: {meta_prompt}", image=processed_image)
        final_plan_raw = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=self.meta_prompt_inspect.format(action_space=self.action_space)
        )
        self.log(name="Final raw plan", log_type="info", message=plan_raw)

        plan_code, filtered_masks = self.extract_plans_and_regions(final_plan_raw, masks)

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


def agent_factory(agent_type, segmentor=None, vlm=None, detector=None, llm=None, configs=None, logger=None):
    """
    Factory method to create an instance of a specific Agent subclass with default values.

    Args:
        agent_type (str): The type of agent to create. Possible values are 'SegVLM', 'DetVLM', and 'DetLLM'.
        segmentor (Segmentor, optional): An instance of Segmentor. Defaults to a default instance if not provided.
        vlm (LanguageModel, optional): An instance of LanguageModel for VLM. Defaults to a default instance if not provided.
        detector (Detector, optional): An instance of Detector. Defaults to a default instance if not provided.
        llm (LanguageModel, optional): An instance of LanguageModel for LLM. Defaults to a default instance if not provided.
        configs (dict, optional): A dictionary of configuration settings.

    Returns:
        Agent: An instance of the specified Agent subclass.
    """

    # Use default instances if none are provided
    from apis.detectors import OWLViT
    from apis.segmentors import SAM
    from apis.language_model import GPT4, GPT4V
    segmentor = segmentor or SAM()
    vlm = vlm or GPT4V()
    detector = detector or OWLViT()
    llm = llm or GPT4()

    if agent_type == 'SegVLM':
        return SegVLM(segmentor=segmentor, vlm=vlm, configs=configs, logger=logger)

    elif agent_type == 'DetVLM':
        return DetVLM(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs, logger=logger)

    elif agent_type == 'DetLLM':
        return DetLLM(segmentor=segmentor, detector=detector, llm=llm, configs=configs, logger=logger)

    elif agent_type == 'VLMSeg':
        return VLMDetInspect(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs, logger=logger)

    elif agent_type == 'VLMDet':
        return VLMDet(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs, logger=logger)

    elif agent_type == 'VLMDetInspect':
        return VLMDetInspect(segmentor=segmentor, detector=detector, vlm=vlm, configs=configs, logger=logger)
    
    else:
        raise ValueError("Unknown agent type.")
    
