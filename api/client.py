import requests
import time
from typing import Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLInferenceClient:
    """
    The client for sending request to the ML inference server
    """

    def __init__(self, server_host: str = "localhost", server_port: int = 8000):
        self.base_url = f"http://{server_host}:{server_port}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def check_server_status(self) -> Dict:
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to connect to server: {e}"}

    def health_check(self) -> Dict:
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {e}"}

    def predict_single_image(
        self,
        image_path: str,
        target_class_ids: List[int],
        threshold: float,
    ) -> Dict:

        if not Path(image_path).exists():
            return {"error": f"image not found at {image_path}"}
        requests_json = {
            "image_path": image_path,
            "target_class_ids": target_class_ids,
            "threshold": threshold,
        }
        start_time = time.time()
        response = self.session.post(f"{self.base_url}/predict", json=requests_json)
        response.raise_for_status()
        result = response.json()
        request_time = (time.time() - start_time) * 1000
        logger.info(f"Request completed in {request_time:.2f} ms")

        return result


def predict_image(
    image_path: str,
    server_host: str = "localhost",
    server_port: int = 8080,
) -> Dict:
    client = MLInferenceClient(server_host, server_port)
    return client.predict_single_image(image_path=image_path)
