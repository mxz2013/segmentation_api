"""
Server main entry point - Run this on the inference server computer
"""

import argparse
import torch
from segmentation_api.config.configure import VALID_MODEL_NAMES
from segmentation_api.api.server import run_server


def main():
    parser = argparse.ArgumentParser(description="ML Inference Server")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/deeplabv3_mobilenet_v2_1.0_513",
        choices=VALID_MODEL_NAMES,
        help="model name for segmentation",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device for inference (cuda or cpu)"
    )
    args = parser.parse_args()

    if not args.device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 50)
    print("ML INFERENCE SERVER")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print("=" * 50)

    try:
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            model_name=args.model,
            device=device,
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()
