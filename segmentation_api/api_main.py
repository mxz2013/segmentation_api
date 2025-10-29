"""
A single interface for launching server and client operations instead of separate server.py and client.py files.
"""

import argparse
from segmentation_api.api.client import MLInferenceClient
from segmentation_api.api.server import run_server


def main():
    parser = argparse.ArgumentParser(description="ML Inference System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands (server/client)")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the inference server")
    server_parser.add_argument("--model", required=True, help="Model name")
    server_parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")
    server_parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers"
    )

    # Client command
    client_parser = subparsers.add_parser("predict", help="Make a prediction")
    client_parser.add_argument("--image", required=True, help="Path to image")
    client_parser.add_argument("--host", default="localhost", help="Server host")
    client_parser.add_argument("--port", type=int, default=8000, help="Server port")
    client_parser.add_argument(
        "--target_ids", nargs="+", type=int, help="Target class IDs"
    )
    client_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")

    args = parser.parse_args()

    if args.command == "server":
        run_server(args.model, args.device, args.host, args.port, args.workers)
    elif args.command == "client":
        client = MLInferenceClient(args.host, args.port)
        result = client.predict_single_image(args.image, args.classes, args.threshold)
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
