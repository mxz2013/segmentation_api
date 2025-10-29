import argparse
import logging
from segmentation_api.config.configure import DEFAULT_TARGETS
from segmentation_api.api.client import MLInferenceClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ML Inference Client")

    parser.add_argument(
        "--server",
        type=str,
        default="localhost",
        help="Server host/IP (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--target_ids",
        type=int,  # Convert each argument to an integer
        nargs="+",  # Accept one or more values, which will be collected into a list
        default=DEFAULT_TARGETS,
        help="List of target IDs for segmentation (e.g., 1 2 3)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Input image path")

    args = parser.parse_args()

    print("=" * 50)
    print("ML INFERENCE CLIENT")
    print("=" * 50)
    print(f"Server: {args.server}:{args.port}")
    print("=" * 50)
    logger.info(f" Single Image Prediction ")
    logger.info(f"Image {args.image}")
    logger.info(f"Class ID {args.target_ids}")
    logger.info(f"Threshold {args.threshold}")
    # initialize client
    client = MLInferenceClient(args.server, args.port)

    result = client.predict_single_image_async(
        image_path=args.image,
        target_class_ids=args.target_ids,
        threshold=args.threshold,
    )

    print("=" * 50)
    logger.info(f"{result = }")
    return result


if __name__ == "__main__":
    main()
