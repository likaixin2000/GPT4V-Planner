import requests
import pickle
import base64
import numpy as np
from PIL import Image

from .utils import convert_pil_image_to_base64

class Segmentor():
    pass

class SAM(Segmentor):
    server_url = "http://ml4.d2.comp.nus.edu.sg:55563"

    def _send_request(self, endpoint: str, image: Image, additional_data: dict = None):
        """
        Send a request to the server with the specified image and additional data.
        
        :param endpoint: The endpoint for the specific segmentation method.
        :param image: The image to be segmented.
        :param additional_data: Additional data required by the specific method.
        :return: The response from the server.
        """
        image_base64 = convert_pil_image_to_base64(image)
        payload = {"image": image_base64}
        if additional_data:
            payload.update(additional_data)

        # Convert numpy arrays to lists
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()

        response = requests.post(f"{self.server_url}/{endpoint}", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()

    def segment_auto_mask(self, image: Image):
        """
        Automatically generate masks for the image.
        
        :param image: The image to be segmented.
        :return: Segmentation results.
        """
        response = self._send_request('sam_auto_mask_generation', image)
        return self._process_response(response)

    def segment_by_point_set(self, image: Image, points: list, point_labels: list):
        """
        Generate masks for the image based on the provided point set (in 0-1 range).
        
        :param image: The image to be segmented.
        :param points: The points for segmentation (in 0-1 range). Shape: (nb_predictions, nb_points_per_mask, 2).
        :return: Segmentation results.
        """
        scaled_points = self._scale_points_to_image_size(points, image.size)
        # Request points should be in shape (nb_predictions, nb_points_per_mask, 2)

        response = self._send_request(
            endpoint='sam_mask_by_point_set',
            image=image, 
            additional_data={
                'points': scaled_points, 
                'labels': point_labels,
                'return_best': True
            }
        )
        return self._process_response(response)

    def segment_by_bboxes(self, image: Image, bboxes: list):
        """
        Generate masks for the image based on the provided bounding box (in 0-1 range).
        
        :param image: The image to be segmented.
        :param bbox: The bounding box for segmentation (in 0-1 range).
        :return: Segmentation results.
        """
        scaled_bboxes = self._scale_bboxes_to_image_size(bboxes, image.size)
        response = self._send_request(
            endpoint='sam_mask_by_bbox',
            image=image, 
            additional_data={
                'bboxes': scaled_bboxes,
                'return_best': True
            }
        )
        return self._process_response(response)

    def _scale_points_to_image_size(self, points, image_size):
        """
        Scale points from 0-1 range to image size for a 3D array.

        :param points: 3D list of points in 0-1 range (nb_predictions, nb_points_per_mask, 2).
        :param image_size: Size of the image (width, height).
        :return: 3D list of points scaled to the image size.
        """
        width, height = image_size
        scaled_points = []

        for points_set in points:
            scaled_set = [[int(x * width), int(y * height)] for x, y in points_set]
            scaled_points.append(scaled_set)

        return scaled_points

    def _scale_bboxes_to_image_size(self, bboxes, image_size):
        """
        Scale bounding boxes from 0-1 range to image size for a 3D array.

        :param bboxes: 3D list of bounding boxes in 0-1 range (nb_predictions, nb_bboxes_per_mask, 4).
        :param image_size: Size of the image (width, height).
        :return: 3D list of bounding boxes scaled to the image size.
        """
        width, height = image_size
        scaled_bboxes = []

        for bbox_set in bboxes:
            assert len(bbox_set) == 1, "Only one bounding box allowed for each prediction."
            bbox = bbox_set[0]
            scaled_set = [[int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)]]
            scaled_bboxes.append(scaled_set)

        return scaled_bboxes

    def _process_response(self, response):
        """
        Process the response from the server.
        
        :param response: The response from the server.
        :return: Processed segmentation results.
        """
        results = []
        for item in response["result"]:
            tmp_dict = {}
            # Decode the base64 string and then unpickle it
            tmp_dict["segmentation"] = pickle.loads(base64.b64decode(item["segmentation"]))
            del item["segmentation"]
            
            # Convert lists back to numpy arrays if necessary
            for key, value in item.items():
                if isinstance(value, list):
                    tmp_dict[key] = np.array(value)
                else:
                    tmp_dict[key] = value
            results.append(tmp_dict)
        return results
    