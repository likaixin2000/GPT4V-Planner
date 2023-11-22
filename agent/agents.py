
from PIL import Image

from api.language_model import LanguageModel
from api.detectors import Detector
from api.segmentors import Segmentor


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


class Agent():
    pass


class SegVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by thte robot's camera, in which some objects are highlighted with masks and marked with numbers. Output your plan as code.

Operation list:
- pick(item)
- place(item, orientation)
- open(item)


Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
'''

    def __init__(self, 
        segmentor: Segmentor, vlm: 
        LanguageModel,
        configs: dict = None
    ):
        if not isinstance(segmentor, Segmentor):
            raise TypeError("`segmentor` must be an instance of Segmentor.")
        if not isinstance(vlm, LanguageModel):
            raise TypeError("`vlm` must be an instance of LanguageModel.")

        self.segmentor = segmentor
        self.vlm = vlm

        # Configs
        if configs is not None:
            self.configs = configs
        else:
            # Default configs
            self.configs = {
                "img_size": 640,
                "label_mode": "1",
                "alpha": 0.05
            }


    def work(self, prompt: str, image: Image, return_anno_image=False):
        # Resize the image if necessary
        processed_image = image
        if "img_size" in self.configs:
            processed_image = resize_image(image, self.configs["img_size"])
        # Generate segmentation masks
        results = self.segmentor.segment_objects(processed_image)

        # Draw masks
        sorted_anns = sorted(results, key=(lambda x: x['area']), reverse=True)
        annotated_img = visualize_masks(processed_image, 
                            annotations=[anno["segmentation"] for anno in sorted_anns],
                            label_mode=self.configs["label_mode"],
                            alpha=self.configs["alpha"],
                            draw_mask=False, 
                            draw_mark=True, 
                            draw_box=False
        )
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(annotated_img)
        # plt.show()
        
        result = self.vlm.chat(prompt=prompt, image=annotated_img, meta_prompt=self.meta_prompt)
        if return_anno_image:
            return result, annotated_img
        else:
            return result


class DetVLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will see an image captured by thte robot's camera, in which some objects are highlighted with bounding boxes and marked with numbers. Output your plan as code.

Operation list:
- pick(item)
- place(item, orientation)
- open(item)


Note:
- For any item mentioned in your answer, please use the format of `regions[number]`.
'''

    def __init__(
            self, 
            detector: Detector, 
            vlm: LanguageModel,
            configs: dict = None
            ):
        if not isinstance(detector, Detector):
            raise TypeError("`detector` must be an instance of Detector.")
        if not isinstance(vlm, LanguageModel):
            raise TypeError("`vlm` must be an instance of LanguageModel.")

        self.detector = detector
        self.vlm = vlm
        # Configs
        if configs is not None:
            self.configs = configs
        else:
            # Default configs
            self.configs = {
                "img_size": 640,
                "label_mode": "1",
                "alpha": 0.05
            }

    
    def work(self, prompt: str, image: Image, return_anno_image=False):
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

        # Draw masks
        annotated_img = visualize_bboxes(
            image,
            bboxes=[obj['box'] for obj in detected_objects], 
            alpha=self.configs["alpha"]
        )
        
        
        result = self.vlm.chat(prompt=prompt, image=annotated_img, meta_prompt=self.meta_prompt)
        if return_anno_image:
            return result, annotated_img
        else:
            return result


class DetLLM(Agent):
    meta_prompt = \
'''
You are in charge of controlling a robot. You will be given a list of operations you are allowed to perform, along with a task to solve. You will be given a list of objects detected which you may want to interact with. Output your plan as code.

Operation list:
- pick(item)
- place(item, orientation)
- open(item)

Note:
- For any item mentioned in your answer, please use the format of `object_name`.
'''
    def __init__(
            self, 
            detector: Detector, 
            vlm: LanguageModel,
            configs: dict = None
            ):
        if not isinstance(detector, Detector):
            raise TypeError("`detector` must be an instance of Detector.")
        if not isinstance(vlm, LanguageModel):
            raise TypeError("`vlm` must be an instance of LanguageModel.")

        self.detector = detector
        self.vlm = vlm
        # Configs
        if configs is not None:
            self.configs = configs
        else:
            # Default configs
            self.configs = {
                "img_size": 640,
                "label_mode": "1",
                "alpha": 0.05
            }

    
    def work(self, prompt: str, image: Image, return_anno_image=False):
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

        # Draw masks
        annotated_img = visualize_bboxes(
            image,
            bboxes=[obj['box'] for obj in detected_objects], 
            alpha=self.configs["alpha"]
        )
        
        
        result = self.vlm.chat(prompt=prompt, image=annotated_img, meta_prompt=self.meta_prompt)
        if return_anno_image:
            return result, annotated_img
        else:
            return result