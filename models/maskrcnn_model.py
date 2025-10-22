"""
implement a simple Mask R-CNN pretrained model to inference an input image
produce segmentation results for cats
It can be extended to other pretrained model, e.g., YOLO, UNET
"""

import torch
import numpy as np
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
        image_tensor: torch.Tensor,
        target_class_ids: Optional[List[int]] = None,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        run inference and find the results

        :param image_tensor: input image in tensor
        :param target_class_ids: input image in tensor
        :param threshold: user input classification threshold
        :return: Dict with keys: masks, boxes, scores, cat_count, pix_count
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            predictions = self.model([image_tensor])[0]
        # if
        cur_target_ids = target_class_ids or self.default_targe_ids
        cur_threshold = threshold or self.default_threshold
        # we only want cat
        target_index = (predictions["labels"] in cur_target_ids) & (
            predictions["scores"] > cur_threshold
        )
        if not target_index.any():
            return {
                "masks": np.array([]),
                "boxes": np.array([]),
                "scores": np.array([]),
                "cat_count": 0,
                "pix_count": 0,
            }
        # mask in shape N, C, H, W
        masks = predictions["masks"][target_index].cpu().numpy()
        boxes = predictions["boxes"][target_index].cpu().numpy()
        scores = predictions["scores"][target_index].cpu().numpy()

        # binary the mask and count the mask pixels
        masks = (masks > self.confidence_threshold).astype(np.uint8)
        # sum only H, W, squeeze the chanel dim
        pix_count = np.squeeze(np.sum(masks, axis=(2, 3)), axis=1)

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "cat_count": len(scores),
            "pix_count": pix_count,
        }


# class PredictionOutput(BaseModel):
#     masks: np.ndarray
#     boxes: np.ndarray
#     scores: np.ndarray
#     target_count: int
