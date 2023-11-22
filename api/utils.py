import re
import base64
from io import BytesIO

from PIL import Image

def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



def highlight_mask(text, masks):
    # Use regex to find all numbers in '[]'
    res = re.findall(r'\[(\d+)\]', text)
    # Convert extracted strings to integers and remove duplicates
    res = list(set(map(int, res)))

    sections = []
    for i, r in enumerate(res):
        mask_i = masks[0][r - 1]['segmentation']
        sections.append((mask_i, str(r)))
    return (history_images[0], sections)