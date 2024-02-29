import argparse
import base64
from io import BytesIO
import requests

from flask import Flask, request, jsonify
from transformers import pipeline, BitsAndBytesConfig
import torch
from PIL import Image

# LLaVA model setup
model_id = "llava-hf/llava-1.5-13b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# Flask app
app = Flask(__name__)


def convert_base64_to_pil_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return image


@app.route('/llava_chat', methods=['POST'])
def llava_chat():
    # Parse JSON data
    query_data = request.get_json()
    base64_image = query_data["image"]
    prompt = query_data["prompt"]
    max_new_tokens = query_data.get("max_new_tokens", 1024)

    # Convert base64 to PIL Image
    image = convert_base64_to_pil_image(base64_image)

    # Correctly construct the full prompt with placeholders for meta_prompt, image, and prompt
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Generate response
    outputs = pipe(image, prompt=full_prompt, generate_kwargs={"max_new_tokens": max_new_tokens})

    # Return results as JSON
    return jsonify({
        'text': outputs[0]['generated_text']
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=55575, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)
