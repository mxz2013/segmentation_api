import argparse
import logging
import torch
from config.configure import VALID_MODEL_NAMES, DEFAULT_TARGETS, DEFAULT_THRESHOLD
from inference.segmenter import Segmenter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Segmentation Inference")

    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--model",
        type=str,
        default="google/deeplabv3_mobilenet_v2_1.0_513",
        choices=VALID_MODEL_NAMES,
        help="model name for segmentation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="the confidence threshold for mask",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device for inference (cuda or cpu)"
    )

    args = parser.parse_args()

    if not args.device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    segmenter = Segmenter(device=device, model_name=args.model)

    logger.info(f"Processing {args.image}")

    results = segmenter.segment(args.image)
    print(f"{results = }")


if __name__ == "__main__":

    main()
