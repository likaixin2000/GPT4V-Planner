import base64
import io
import os
import sys
sys.path.append('big_vision/')

import json

import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io

from flask import Flask, request, jsonify

# Flask app
app = Flask(__name__)

# Load model
print("Loading model, please wait...")
config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
module = models.TextZeroShotDetectionModule(
    body_configs=config.model.body,
    objectness_head_configs=config.model.objectness_head,
    normalize=config.model.normalize,
    box_bias=config.model.box_bias
)
variables = module.load_variables(config.init_from.checkpoint_path)
jitted = jax.jit(module.apply, static_argnames=('train',))


def predict_boxes(image, text_queries):
    tokenized_queries = np.array([
        module.tokenize(q, config.dataset_configs.max_query_length)
            for q in text_queries
        ]
    )

    tokenized_queries = np.pad(
        tokenized_queries,
        pad_width=((0, 100 - len(text_queries)), (0, 0)),
        constant_values=0
    )
    predictions = jitted(
        variables,
        image[None, ...],
        tokenized_queries[None, ...],
        train=False
    )
    
    # Remove batch dimension and convert to numpy
    predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)

    logits = predictions['pred_logits'][..., :len(text_queries)]  # Remove padding.
    scores = sigmoid(np.max(logits, axis=-1))
    labels = np.argmax(predictions['pred_logits'], axis=-1)
    box_names = np.array(text_queries)[labels]
    boxes = predictions['pred_boxes']
    objectnesses = sigmoid(predictions['objectness_logits'])

    return scores, boxes, box_names, objectnesses

def keep_top_k(*args, top_k, key, ):
    top_k_indices = np.argpartition(key, -top_k)[-top_k:]
    return (array[top_k_indices] for array in args)

def keep_by_threshold(*args, threshold, key, ):
    valid_indices = np.where(key > threshold)[0]
    if valid_indices.size == 0:
        return (np.array([]) for _ in range(len(args)))
    return (array[valid_indices] for array in args)

@app.route('/owl_detect', methods=['POST'])
def detect_objects():
    # Parse JSON data
    query_data = request.get_json()
    text_queries = query_data["text_queries"]
    base64_image = query_data["image"]

    # Decode the base64 image
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    
    # Convert image to compatible format
    image_uint8 = skimage_io.imread(image_data)
    image = image_uint8.astype(np.float32) / 255.0

    # Pre-processing
    # Pad to square with gray pixels on bottom and right:
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5
    )

    # Resize to model input size:
    input_image = skimage.transform.resize(
        image_padded,
        (config.dataset_configs.input_size, config.dataset_configs.input_size),
        anti_aliasing=True
    )

    scores, boxes, box_names, objectnesses = predict_boxes(input_image, text_queries)

    # Postprocess
    if "bbox_score_top_k" in query_data:
        scores, boxes, box_names, objectnesses = keep_top_k(
            scores, boxes, box_names, objectnesses,
            top_k=query_data["bbox_score_top_k"],
            key=scores,
        )

    if "bbox_conf_threshold" in query_data:
        scores, boxes, box_names, objectnesses = keep_by_threshold(
            scores, boxes, box_names, objectnesses,
            threshold=query_data["bbox_conf_threshold"],
            key=scores
        )
    
    # Return results as JSON
    return jsonify({
        'scores': scores.tolist(), 
        'bboxes': boxes.tolist(), 
        'box_names': box_names.tolist(),
        'objectnesses': objectnesses.tolist()
    })

if __name__ == '__main__':
    ip = "ml4.d2.comp.nus.edu.sg"  # Use "0.0.0.0" to run on your machine's IP address
    port = 55570     # You can change this to any desired port
    app.run(host=ip, port=port, debug=True, use_reloader=False)