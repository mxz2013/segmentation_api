from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SegmentationBaseModel(ABC):
    """
    An abstract class for segmentation models
    """

    def __init__(self, device: str, default_threshold: float = 0.5):
        self.device = device
        self.default_threshold = default_threshold
        self.model = None

    @abstractmethod
    def load_model(self):
        """
        load a pretrained  model
        :return:
        """
        pass

    @abstractmethod
    def predict(self, image: Image) -> Dict:
        """
        run an inference on an image
        :param image:
        :return: a dict with target_id: pix_count
        """
        pass
