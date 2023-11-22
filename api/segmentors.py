import requests
import pickle
import base64
import numpy as np
from PIL import Image

from .utils import convert_pil_image_to_base64

class Segmentor():
    pass

class SAM(Segmentor):
    server_url = "http://ml4.d2.comp.nus.edu.sg:55563/sam_inference"

    def segment_objects(self, image: Image):
        # transform = transforms.Compose([transforms.Resize(int(img_size), interpolation=Image.BICUBIC)])
        # image_ori = transform(image)
        image_base64 = convert_pil_image_to_base64(image)
        payload = {
            "image": image_base64,
        }

        response = requests.post(self.server_url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        results = []
        for item in response.json()["result"]:
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
        