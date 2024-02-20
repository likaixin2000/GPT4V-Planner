import pytest
from PIL import Image
from apis import GEMINI_PRO, GEMINI_PRO_VISION


@pytest.fixture
def setup_gemini_pro():
    """Fixture to setup GEMINI_PRO instance."""
    gemini_pro = GEMINI_PRO(temperature=0.5)
    return gemini_pro

@pytest.fixture
def setup_gemini_pro_vision():
    """Fixture to setup GEMINI_PRO_VISION instance."""
    gemini_pro_vision = GEMINI_PRO_VISION(temperature=0.5)
    return gemini_pro_vision

def test_gemini_pro_chat(setup_gemini_pro):
    """Test the chat function of GEMINI_PRO."""
    prompt = "Say Hi to me!"
    response = setup_gemini_pro.chat(prompt)
    print("\nGemini Pro responded:\n", response, '\n', '_'*50)
    assert isinstance(response, str)  # Basic check to ensure a string is returned

def test_gemini_pro_vision_chat(setup_gemini_pro_vision):
    """Test the chat function of GEMINI_PRO_VISION with an image."""
    prompt = "What is in the image?"
    image_path = "tests/assets/images/table_top.png"
    image = Image.open(image_path)
    response = setup_gemini_pro_vision.chat(prompt, image)
    print("\nGemini Pro Vision responded:\n", response, '\n', '_'*50)
    assert isinstance(response, str)