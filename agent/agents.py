import re
from PIL import Image

from api.language_model import LanguageModel
from api.detectors import Detector
from api.segmentors import Segmentor

from .visualizer import GenericMask

from .utils import resize_image, visualize_bboxes, visualize_masks

COMMON_OBJECTS = [
    "refrigerator",
    "oven",
    "microwave",
    "toaster",
    "blender",
    "coffee maker",
    "dishwasher",
    "pot",
    "pan",
    "cutting board",
    "knife",
    "spoon",
    "fork",
    "plate",
    "bowl",
    "cup",
    "coaster",
    "glass",
    "kettle",
    "paper towel holder",
    "trash can",
    "food storage container",
    "sofa",
    "coffee table",
    "television",
    "bookshelf",
    "armchair",
    "floor lamp",
    "rug",
    "picture frame",
    "curtain",
    "blanket",
    "vase",
    "indoor plant",
    "remote control",
    "candle",
    "wall art",
    "clock",
    "magazine rack",
]

DEFAULT_ACTION_SPACE = """
 - pick(item)
 - place(item, orientation)
 - open(item)
"""


def extract_plans_and_regions(text: str, regions: list):
    # Extract code blocks. We assume there is only one code block in the generation
    code_block = re.findall(r'```python(.*?)```', text, re.DOTALL)[0]

    # Use regular expression to find all occurrences of region[index]
    matches = re.findall(r'regions\[(\d+)\]', code_block)

    used_indices = list(set(int(index) for index in matches))
    used_indices.sort()

    index_mapping = {old_index: new_index for new_index, old_index in enumerate(used_indices)}
    for old_index, new_index in index_mapping.items():
        text = text.replace(f'regions[{old_index}]', f'regions[{new_index}]')

    filtered_regions = [regions[index] for index in used_indices]

    return text, filtered_regions


class Agent():
    def __init__(self, action_space: str = DEFAULT_ACTION_SPACE) -> None:
        self.action_space = action_space


class SegVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by thte robot's camera, in which some objects are highlighted with masks and marked with numbers. Output your plan as code.

Operation list:
{action_space}

Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
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
            "img_size": 640,
            "label_mode": "1",
            "alpha": 0.05
        }
        if configs is not None:
            self.configs = self.configs.update(configs)  

        super().__init__(**kwargs)


    def plan(self, prompt: str, image: Image, return_info=False):
        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs:
            processed_image = resize_image(image, self.configs["img_size"])
        # Generate segmentation masks
        masks = self.segmentor.segment_auto_mask(processed_image)

        # Draw masks
        # sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        annotated_img = visualize_masks(processed_image, 
                            annotations=[anno["segmentation"] for anno in masks],
                            label_mode=self.configs["label_mode"],
                            alpha=self.configs["alpha"],
                            draw_mask=False, 
                            draw_mark=True, 
                            draw_box=False
        )
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(annotated_img)
        # plt.show()
        
        raw_plan = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=self.meta_prompt.format(action_space=self.action_space)
        )
        
        plan, filtered_masks = extract_plans_and_regions(raw_plan, masks)

        if return_info:
            info_dict = dict(annotated_img=annotated_img, raw_plan=raw_plan)
            return plan, filtered_masks, info_dict
        else:
            return plan, filtered_masks


class DetVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by thte robot's camera, in which some objects are highlighted with bounding boxes and marked with numbers. Output your plan as code.

Operation list:
{action_space}


Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
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
            "img_size": 640,
            "label_mode": "1",
            "alpha": 0.05
        }
        if configs is not None:
            self.configs = self.configs.update(configs)            

        super().__init__(**kwargs)
    
    def plan(self, prompt: str, image: Image, return_info=False):
        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs:
            processed_image = resize_image(image, self.configs["img_size"])
        
        # Generate detection boxes
        text_queries = COMMON_OBJECTS
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


        # Draw masks
        annotated_img = visualize_bboxes(
            image,
            bboxes=[obj['box'] for obj in detected_objects], 
            alpha=self.configs["alpha"]
        )
        
        
        raw_plan = self.vlm.chat(
            prompt=prompt, 
            image=annotated_img, 
            meta_prompt=self.meta_prompt.format(action_space=self.action_space)
        )
        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[[bbox] for bbox in detected_objects])

        plan, filtered_masks = extract_plans_and_regions(raw_plan, masks)

        if return_info:
            info_dict = dict(annotated_img=annotated_img, detected_objects=detected_objects, raw_plan=raw_plan)
            return plan, filtered_masks, info_dict
        else:
            return plan, filtered_masks


class DetLLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will be given a list of objects detected which you may want to interact with. Output your plan as code.

Operation list:
{action_space}

Note:
- For any item mentioned in your answer, please use the format of `"object_name"`.
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
            "img_size": 640,
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
                box = obj['box']
                box_coords = f" (coordinates: ({box[0]:.2f}, {box[1]:.2f}), ({box[2]:.2f}, {box[3]:.2f}))"
                markdown_list.append(f"- {box_name}{box_coords}")
            else:
                markdown_list.append(f"- {box_name}")

        return '\n'.join(markdown_list)

    def plan(self, prompt: str, image: Image, return_info=False):
        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs:
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

        # Covert detection results to a string
        textualized_object_list = self.textualize_detections(detected_objects, include_coordinates=self.configs["include_coordinates"])
        prompt = textualized_object_list + '\n\n' + prompt
        
        raw_plan = self.llm.chat(
            prompt=prompt, 
            meta_prompt=self.meta_prompt.format(action_space=self.action_space)
        )

        masks = self.segmentor.segment_by_bboxes(image=image, bboxes=[[bbox] for bbox in detected_objects])

        plan, filtered_masks = extract_plans_and_regions(raw_plan, masks)

        if return_info:
            info_dict = dict(detected_objects=detected_objects, raw_plan=raw_plan)
            return plan, filtered_masks, info_dict
        else:
            return plan, filtered_masks


def agent_factory(agent_type, segmentor=None, vlm=None, detector=None, llm=None, configs=None):
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
    from api.detectors import OWLViT
    from api.segmentors import SAM
    from api.language_model import GPT4, GPT4V
    segmentor = segmentor or SAM()
    vlm = vlm or GPT4V()
    detector = detector or OWLViT()
    llm = llm or GPT4()

    if agent_type == 'SegVLM':
        return SegVLM(segmentor=segmentor, vlm=vlm, configs=configs)

    elif agent_type == 'DetVLM':
        return DetVLM(segmentor=segmentor,detector=detector, vlm=vlm, configs=configs)

    elif agent_type == 'DetLLM':
        return DetLLM(segmentor=segmentor,detector=detector, llm=llm, configs=configs)

    else:
        raise ValueError("Unknown agent type.")
    