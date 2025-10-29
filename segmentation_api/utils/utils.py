from PIL import Image
import logging

logger = logging.getLogger(__name__)


def process_image(image_path: str) -> Image:
    """
    load and preprocess an input image
    TODO: handel image with different sizes by introducing min_size, and max_size parameters
    :param image_path: image path
    :return: (PIL Image, torch Tensor)
    """
    logger.info(f"Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")
    return image
