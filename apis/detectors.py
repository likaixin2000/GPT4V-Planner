import json
from PIL import Image

import requests

from utils.image_utils import convert_pil_image_to_base64

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
    "phone",
    "pen",
    "marker",
    "laptop",
    "tape",
    "keyboard",
    "block",
]

# def convert_cxcywh_to_x1y1x2y2(cxcywh_box):
#     """
#     Converts bounding box coordinates from cXcYWH format to X1Y1X2Y2 format.

#     Parameters:
#     xywh_box (list or tuple): Bounding box in XYWH format [cx, cy, width, height].

#     Returns:
#     list: Bounding box in X1Y1X2Y2 format [x1, y1, x2, y2].
#     """
#     cx, cy, w, h = cxcywh_box
#     return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

class Detector():
    pass

class OWLViT(Detector):

    def __init__(self, server_url="http://phoenix0.d2.comp.nus.edu.sg:55570/owl_detect"):
        self.server_url = server_url

    def detect_objects(self, image: Image.Image, text_queries: list[str], bbox_score_top_k=20, bbox_conf_threshold=0.5):
        """
        Function to call an object detection API and return the response.

        Parameters:
        - api_url (str): URL of the object detection API.
        - text_queries (list of str): Text queries for object detection.
        - image_file_path (str): File path to the image to be analyzed.
        - bbox_score_top_k (int, optional): Number of top scoring bounding boxes to return. Defaults to 20.
        - bbox_conf_threshold (float, optional): Confidence threshold for bounding boxes. Defaults to 0.5.
        
        Returns:
        - tuple: Parsed response data from the API, containing scores, boxes, box_names, and objectnesses.
         Example result:
        [
            {'score': 0.3141017258167267,
            'bbox': [0.212062269449234,
            0.3956533372402191,
            0.29010745882987976,
            0.08735490590333939],
            'box_name': 'roof',
            'objectness': 0.09425540268421173
            }, ...
        ]
        """

        # Convert image to base64
        img_b64_str = convert_pil_image_to_base64(image.convert('RGB'))
        # Constructing the POST request
        payload = {
            "text_queries": text_queries,
            "image": img_b64_str,
            "bbox_score_top_k": bbox_score_top_k,
            "bbox_conf_threshold": bbox_conf_threshold
        }
        response = requests.post(
            self.server_url, 
            json=payload,
        )
        
        # Check for request failure
        if response.status_code != 200:
            raise ConnectionError(f"Request failed with status code {response.status_code}")

        resp_data = json.loads(response.text)
        
        # Retrieve the relevant data
        scores = resp_data['scores']
        bboxes = resp_data['bboxes']
        box_names = resp_data['box_names']
        
        
        # Assert that all lists have the same length
        assert len({len(scores), len(bboxes), len(box_names)}) == 1, "Server returned data with different lengths. Something is wrong, most probably on the server side."

        dict_data = [{'score': score, 'bbox': bbox, 'box_name': box_name} 
                 for score, bbox, box_name in zip(scores, bboxes, box_names)]

        return dict_data

