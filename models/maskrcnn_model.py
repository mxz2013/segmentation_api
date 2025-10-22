import torch
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from typing import Dict, Optional, List
import logging
from config.configure import DEFAULT_TARGETS, DEFAULT_THRESHOLD

from models.base_model import SegmentationBaseModel

logger = logging.getLogger(__name__)


class MaskRCNNModel(SegmentationBaseModel):
    """
    the mask R-CNN segmentation pretrained model on COCO dataset from pytorch
    """

    def __init__(
        self,
        device: str,
        default_target_ids: List[int] = DEFAULT_TARGETS,
        default_threshold: float = DEFAULT_THRESHOLD,
    ):
        super().__init__(device, default_threshold)
        self.default_targe_ids = default_target_ids
        self.default_threshold = default_threshold
        self.load_model()

    def load_model(self):
        """
        load the pretrained mask r-cnn model
        :param self:
        :return:
        """

        logger.info("Loading Mask R-CNN model ...")

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Mask R-CNN model loaded successfully!")

    def predict(
        self,
        image: Image,
        target_class_ids: Optional[List[int]] = None,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        run inference and find the results

        :param image: input image in PIL.Image format
        :param target_class_ids: input image in tensor
        :param threshold: user input classification threshold
        :return: Dict: target_id: pix_count
        """
        image_tensor = F.to_tensor(image)
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            predictions = self.model([image_tensor])[0]
        # if
        cur_target_ids = target_class_ids or self.default_targe_ids
        cur_threshold = threshold or self.default_threshold
        # get labes in the target ids and classification confidence > 0.7, hardcoded for now
        target_index = np.isin(predictions["labels"], cur_target_ids) & (
            predictions["scores"].cpu().numpy() > 0.7
        )
        if not target_index.any():
            results = {}
            for label in cur_target_ids:
                results[label] = 0
            return results

        # mask in shape N, C, H, W
        labels = predictions["labels"][target_index].cpu().numpy()
        masks = predictions["masks"][target_index].cpu().numpy()

        # binary the mask and count the mask pixels
        masks = (masks > cur_threshold).astype(np.uint8)
        # sum only H, W, squeeze the chanel dim
        pix_count = np.squeeze(np.sum(masks, axis=(2, 3)), axis=1)
        results = {}
        for i, label in enumerate(labels):
            if label not in results:
                results[label] = pix_count[i]
            else:
                results[label] += pix_count[i]

        return results
