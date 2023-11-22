import numpy as np

from PIL import Image
from torchvision import transforms

from .visualizer import Visualizer

def resize_image(image: Image, size) -> Image:
    transform = transforms.Compose([transforms.Resize(int(size), interpolation=Image.BICUBIC)])
    processed_image = transform(image)
    return image

def visualize_bboxes(image: Image, bboxes: list, alpha=0.7) -> Image:
    if len(bboxes) == 0:
        return image
    
    visualizer = Visualizer(image, metadata=None)
    width, height = image.size

    for i, bbox in enumerate(bboxes):
        label = i + 1
        # color_mask = np.random.random((1, 3)).tolist()[0]
        demo = visualizer.draw_bbox_with_number(bbox, text=str(label), alpha=alpha)
    return Image.fromarray(demo.get_image())


def visualize_masks(image: Image, annotations: list, label_mode, alpha, draw_mask=False, draw_mark=True, draw_box=False) -> Image:
    if len(annotations) == 0:
        return image

    visualizer = Visualizer(image, metadata=None)
    width, height = image.size
    # Construct annotation (drawing) mode
    anno_mode = []
    if draw_mask:
        anno_mode.append("Mask")
    if draw_mark:
        anno_mode.append("Mark")
    if draw_box:
        anno_mode.append("Box")

    mask_map = np.zeros((height, width), dtype=np.uint8)    
    for i, mask in enumerate(annotations):
        label = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        # color_mask = [int(c*255) for c in color_mask]
        demo = visualizer.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        # assign the mask to the mask_map
        mask_map[mask == 1] = label
    return Image.fromarray(demo.get_image())   