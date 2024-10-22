from typing import Optional, Tuple, Union
from typing import List as list
from PIL import Image
import numpy as np
import cv2

def scale_ratio_box_to_pixel(x1, y1, x2, y2, *, width, height):
    x1 *= width
    y1 *= height
    x2 *= width
    y2 *= height
    return x1, y1, x2, y2


def scale_pixel_box_to_ratio(x1, y1, x2, y2, *, width, height):
    x1 /= width
    y1 /= height
    x2 /= width
    y2 /= height
    return x1, y1, x2, y2


class Mask:
    def __init__(self, mask, name=None, identifier=None, ref_image=None):
        self.mask = mask
        self.name = name
        self.identifier = identifier
        self.ref_image = ref_image

    @classmethod
    def from_dict(cls, mask_dict: dict):
        return cls(
            mask=mask_dict["segmentation"]
        )
    
    @classmethod
    def from_list(cls, mask_list: list[dict], ref_image: Image.Image, names: list[str]=None):
        if names:
            assert len(mask_list) == len(names)

        results = []
        for i in range(len(mask_list)):
            name = names[i] if names else None
            results.append(
                cls(
                    mask=mask_list[i],
                    name=name,
                    identifier=i + 1,
                    ref_image=ref_image
                )
            )
        
        return results
    
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        # if not isinstance(self.mask, np.ndarray):
        #     self.mask = np.array(self.mask)
            
        # Find indices where mask is True
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None  # No bounding box if the mask is entirely False
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Convert to ratio
        height, width = self.mask.shape
        x_min, y_min, x_max, y_max = scale_pixel_box_to_ratio(x_min, y_min, x_max, y_max, width=width, height=height)
        
        return (x_min, y_min, x_max, y_max)

    def crop_obj(self, padding: Union[int, float]=0) -> Optional[Image.Image]:
        bbox = self.bbox()
        if self.ref_image is None:
            return None
        
        width, height = self.ref_image.size  # self.ref_image is a PIL Image. Note the order!
        x_min, y_min, x_max, y_max = bbox
        # Scale to pixel space
        x_min, y_min, x_max, y_max = scale_ratio_box_to_pixel(x_min, y_min, x_max, y_max, width=width, height=height)

        # Determine padding type and calculate accordingly
        if isinstance(padding, int):  # pad_pixel
            pad_x = pad_y = padding
        elif isinstance(padding, float):  # pad_ratio
            pad_x = int((x_max - x_min) * padding)
            pad_y = int((y_max - y_min) * padding)
        else:
            raise ValueError("Padding must be an integer (pad_pixel) or a float (pad_ratio).")
        
        # Apply padding and ensure within image bounds
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(width, x_max + pad_x)
        y_max = min(height, y_max + pad_y)
        
        cropped_image = self.ref_image.crop((x_min, y_min, x_max, y_max))  # (left, upper, right, lower)
        cropped_bbox = scale_pixel_box_to_ratio(x_min, y_min, x_max, y_max, width=width, height=height)
        return cropped_image, cropped_bbox
    
    # def _reidentify_using_image(self, new_image: Image, place_point: list[float], detector, segmentor):
    #     # This method does not work for harder queries because of excessive noise.
    #     # See https://discuss.huggingface.co/t/owlv2-image-guided-detection-embed-image-query-why-choosing-the-least-similar-box-from-selected-ones/63390
    #     if self.ref_image is None:
    #         raise ValueError("Reference image is not set.")

    #     query_image, _ = self.crop_obj()
    #     matches = detector.match_by_image(image=new_image, query_image=query_image, match_threshold=0.8, nms_threshold=0.7)

    #     # Initialize variables to find the closest match
    #     closest_match = None
    #     min_distance = float('inf')

    #     for match in matches:
    #         bbox = match["bbox"]
    #         # Calculate the center point of the match
    #         center_x = (bbox[0] + bbox[2]) / 2
    #         center_y = (bbox[1] + bbox[3]) / 2

    #         # Calculate the Euclidean distance to the place_point
    #         distance = np.sqrt((center_x - place_point[0])**2 + (center_y - place_point[1])**2)

    #         # Update if this match is closer
    #         if distance < min_distance:
    #             min_distance = distance
    #             closest_match = bbox

    #     if closest_match is None:
    #         raise RuntimeError("Failed to reidentify the object.")  # If no matches are found or closest match cannot be determined

    #     self.ref_image = new_image
    #     # Update mask
    #     self.mask = segmentor.segment_by_bboxes(new_image, bboxes=[closest_match])[0]["segmentation"]

    def find_mask_center_point(self):
        binary_mask = self.mask
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255  # Make cv2 happy
        height, width = binary_mask.shape
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)
        # Only take one
        coords_x = coords_x[0]
        coords_y = coords_y[0]
    
        return coords_x / width, coords_y / height

    def reidentify(self, new_image: Image, place_point: list[float], detector, segmentor):
        if self.ref_image is None:
            raise ValueError("Reference image is not set.")

        print(f"reidentify: {self.name}")
        matches = detector.detect_objects(
            image=new_image,
            text_queries=[self.name],
            bbox_score_top_k=20,
            bbox_conf_threshold=0.15
        )
        print(f"matches: {matches}")

        # Initialize variables to find the closest match
        closest_match = None
        min_distance = float('inf')

        for match in matches:
            bbox = match["bbox"]
            # Calculate the center point of the match
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Calculate the Euclidean distance to the place_point
            distance = np.sqrt((center_x - place_point[0])**2 + (center_y - place_point[1])**2)

            # Update if this match is closer
            if distance < min_distance:
                min_distance = distance
                closest_match = bbox

        if closest_match is None:
            raise RuntimeError("Failed to reidentify the object.")  # If no matches are found or closest match cannot be determined

        self.ref_image = new_image
        # Update mask
        self.mask = segmentor.segment_by_bboxes(new_image, bboxes=[closest_match])[0]["segmentation"]
