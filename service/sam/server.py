import base64
import io
import pickle
import numpy as np

from PIL import Image
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from flask import Flask, request, jsonify


app = Flask(__name__)


sam_ckpt = "./sam_vit_h_4b8939.pth"
model = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()

def inference_sam_m2m_auto(model, image):
    image_ori = np.asarray(image)

    mask_generator = SamAutomaticMaskGenerator(model)
    outputs = mask_generator.generate(image_ori)
    return outputs

    
@app.route('/sam_inference', methods=['POST'])
def inference():
    data = request.json
    image_base64 = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

    results = inference_sam_m2m_auto(model, image)

    # Compress the mask into pickle object
    for item in results:
        item["segmentation"] = base64.b64encode(pickle.dumps(item["segmentation"])).decode('utf-8')
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                item[key] = value.tolist()

    return jsonify({'result': results})

if __name__ == '__main__':
    # Set custom IP and port
    ip = "ml4.d2.comp.nus.edu.sg"  # Use "0.0.0.0" to run on your machine's IP address
    port = 55563     # You can change this to any desired port
    app.run(host=ip, port=port, debug=True, use_reloader=False)