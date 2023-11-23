import base64
import io
import pickle
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import SamModel, SamProcessor, pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Reference: https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
# For automatic mask generation
generator = pipeline(
    "mask-generation",
    model=model, 
    image_processor=processor.image_processor,
    device=device
)

def process_masks_for_response(masks: list):
    """Process masks for response. Compress masks as numpy array."""
    result = []
    for mask in masks:
        item = {}
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        item["segmentation"] = base64.b64encode(pickle.dumps(mask)).decode('utf-8')
        result.append(item)

    return result

def take_best_masks(masks: list):
    """For each prediction, keep only the best mask."""
    result = [mask[0] for mask in masks]  # The first dimension is the number of mask generated per prediction
    return result

    
@app.route('/sam_auto_mask_generation', methods=['POST'])
def sam_auto_mask_generation():
    data = request.json
    image_base64 = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

    # Call SAM to generate masks
    outputs = generator(image, points_per_batch=15)
    masks = outputs["masks"]

    results = process_masks_for_response(masks)
    return jsonify({'result': results, 'type': 'automatic mask generation'})


@app.route('/sam_mask_by_point_set', methods=['POST'])
def sam_mask_by_point_set():
    data = request.json
    image_base64 = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    points = data['points']  # [[[550, 600], [2100, 1000]]]
    labels = data['labels']
    return_best = data['return_best']

    # One image per batch, thus the added dimension.
    inputs = processor(image, input_points=[points], input_labels=[labels], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            multimask_output=not return_best,
            **inputs
        )

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )[0]  # Only allow one image
    if return_best:
        masks = take_best_masks(masks)
    scores = outputs.iou_scores[0]

    results = process_masks_for_response(masks)
    return jsonify({'result': results, 'type': 'mask by point set'})


@app.route('/sam_mask_by_bbox', methods=['POST'])
def sam_mask_by_bbox():
    data = request.json
    image_base64 = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    bboxes = data['bboxes']  # [[[650, 900, 1000, 1250]]]
    return_best = data['return_best']

    # One image per batch, thus the added dimension.
    inputs = processor(image, input_boxes=[bboxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            multimask_output=not return_best,
            **inputs
        )

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )[0]  # Only allow one image
    if return_best:
        masks = take_best_masks(masks)
    scores = outputs.iou_scores[0]

    results = process_masks_for_response(masks)
    return jsonify({'result': results, 'type': 'mask by bounding box'})


if __name__ == '__main__':
    # Set custom IP and port
    ip = "ml4.d2.comp.nus.edu.sg"  # Use "0.0.0.0" to run on your machine's IP address
    port = 55563     # You can change this to any desired port
    app.run(host=ip, port=port, debug=True, use_reloader=False)