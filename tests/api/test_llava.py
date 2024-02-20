import pytest
import base64
from PIL import Image
import json
from io import BytesIO

from apis.language_model import LLaVA

@pytest.fixture
def client():
    client = LLaVA()
    return client

def test_llava_chat(client):
    """Test the LLaVA chat function with an image and a prompt."""
    prompt = "What is in the image?"
    image = Image.open("tests/assets/images/table_top.png")  # Adjust path as necessary
    response = client.chat(prompt, image)
    assert isinstance(response, str)  # Ensure the response is a string

    print("\nLLaVA responded:\n", response, '\n', '_'*50)
