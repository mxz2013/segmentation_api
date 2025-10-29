import torch
import numpy as np
from typing import Dict, Optional, List
import logging
from segmentation_api.config.configure import DEFAULT_TARGETS, DEFAULT_THRESHOLD
from segmentation_api.models.base_model import SegmentationBaseModel
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

logger = logging.getLogger(__name__)


class HFSementicSegmentation(SegmentationBaseModel):
    """ """

    def __init__(
        self,
        device: str,
        default_target_ids: List[int] = DEFAULT_TARGETS,
        default_threshold: float = DEFAULT_THRESHOLD,
        model_name: str = "google/deeplabv3_mobilenet_v2_1.0_513",
    ):
        super().__init__(device, default_threshold)
        self.model_name = model_name
        self.default_targe_ids = default_target_ids
        self.default_threshold = default_threshold
        self.preprocessor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self.load_model()

    def load_model(self):
        """
        load the pretrained mask r-cnn model
        :param self:
        :return:
        """

        logger.info(f"Loading {self.model_name} ...")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f" {self.model_name} model loaded successfully!")

    def predict(
        self,
        image: Image,
        target_class_ids: Optional[List[int]] = None,
        threshold: Optional[List[int]] = None,
    ) -> Dict:
        """

        :param image:
        :param target_class_ids:
        :param threshold:
        :return:
        """
        with torch.no_grad():
            inputs = self.preprocessor(images=image, return_tensors="pt")
            # TODO check how to take into account the user defined threshold for the mask
            # cur_threshold = threshold or self.default_threshold
            outputs = self.model(**inputs)
            predicted_mask = self.preprocessor.post_process_semantic_segmentation(
                outputs
            )
            # 65 * 65 array, with cat = 8
            mask_numpy = predicted_mask[0].numpy()
            mask_pil = Image.fromarray(mask_numpy.astype(np.uint8))
            target_size = (image.size[0], image.size[1])
            mask_resized_pil = mask_pil.resize(
                target_size, resample=Image.Resampling.NEAREST
            )
            mask_orig_dim = np.array(mask_resized_pil)
        # use user input from the client if target_class_ids else use the default ones
        cur_target_ids = target_class_ids or self.default_targe_ids

        pixel_counts_dict = {}

        for target_id in cur_target_ids:
            is_match = mask_orig_dim == target_id
            count = np.sum(is_match)
            pixel_counts_dict[target_id] = int(count)

        return pixel_counts_dict
