import sys
sys.path.append('../')
import os
import json
import requests

import numpy as np
from PIL import Image

from api import detect_objects

def test_detect_objects():
    # API endpoint URL
    url = "http://127.0.0.1:8890/detect_objects"
 
    # Example text queries
    text_queries = ['apple', 'rocket', 'nasa badge', 'star-spangled banner']
    
    # Example image file
    file_path = '/home/kaixin/vision_feedback/test.png'
    image = Image.open(file_path)
    scores, boxes, box_names, objectnesses = detect_objects(text_queries, image, 
                                                            bbox_score_top_k=20,
                                                            bbox_conf_threshold=0.01,
                                                            api_url=url)

    # Iterate and print each value
    for i in range(len(boxes)):
        print(f"--- Detection {i+1} ---")
        print(f"Box Name: {box_names[i]}")
        print(f"Score: {scores[i]}")
        print(f"Box Coordinates: {boxes[i]}")
        print(f"Objectness: {objectnesses[i]}\n")

# Run the test function
if __name__ == "__main__":
    test_detect_objects()