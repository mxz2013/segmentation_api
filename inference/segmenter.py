from typing import Dict, Optional, List
import logging
from models.maskrcnn_model import MaskRCNNModel
from models.huggingface_sementic_segmentation_model import HFSementicSegmentation
from utils.utils import process_image
from config.configure import VALID_MODEL_NAMES


logger = logging.getLogger(__name__)


class Segmenter:
    """
    segmentation class
    """

    def __init__(
        self, device: str, model_name: str = "google/deeplabv3_mobilenet_v2_1.0_513"
    ):

        self.device = device
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self):
        if self.model_name in VALID_MODEL_NAMES:
            if self.model_name == "mask_rcnn":
                return MaskRCNNModel(device=self.device)
            # TODO add other segmentation models
            elif self.model_name == "google/deeplabv3_mobilenet_v2_1.0_513":
                return HFSementicSegmentation(device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def segment(
        self,
        image_path: str,
        target_class_ids: Optional[List[int]] = None,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        :param image_path: input image path
        :param target_class_ids: list of target class ids to segment, if None, use default, otherwise user defined
        :param threshold: the threshold for binary classification, if None, use default, otherwise user defined
        :return:
        """

        try:
            image = process_image(image_path)
            results = self.model.predict(image, target_class_ids, threshold)
            return results

        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}: {str(e)}")
            raise
