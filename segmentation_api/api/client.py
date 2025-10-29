# asynchronous job polling and result retrieval

import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from segmentation_api.api.schemas import PredictionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLInferenceClient:
    """
    The ML client for sending request to the ML inference server
    """

    def __init__(self, server_host: str = "localhost", server_port: int = 8000):
        self.base_url = f"http://{server_host}:{server_port}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def check_server_status(self) -> Dict:
        """
        Check the status of the server.
        Returns:
            Dictionary with server status.
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking server status: {e}")
            return {"status": "error", "message": str(e)}

    def _poll_job_status(
        self, job_id: str, poll_interval: float = 1.0, max_tries: int = 60
    ) -> Dict[str, Any]:
        """
        Poll the job status from the server until completion or failure.

        Args:
            job_id:
            poll_interval:
            max_tries:

        Returns:

        """

        for i in range(max_tries):
            time.sleep(poll_interval)
            try:
                response = self.session.get(f"{self.base_url}/job_status/{job_id}")
                response.raise_for_status()
                status_data = response.json()
                if status_data.get("status") == "finished":
                    logger.info(f"Job {job_id} complete successfully.")
                    return status_data

                if status_data.get("status") == "failed":
                    logger.error(
                        f"Job {job_id} failed with error: {status_data.get('error_message')}"
                    )
                    return status_data

                logger.info(
                    f"Job {job_id} status: {status_data.get('status')}. Polling again... ({i+1}/{max_tries})"
                )

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # If the job was processed and deleted from Redis, the server returns 404.
                    # We check if the response body contains a result structure.
                    try:
                        status_result = e.response.json()
                        if status_result.get("result"):
                            return status_result["result"]
                    except:
                        pass
                    return {
                        "error": f"Job ID {job_id} not found or result already retrieved."
                    }
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Error during polling job {job_id}: {e}")
                break

        return {
            "error": f"Job {job_id} timed out after {max_tries} tries or failed to connect."
        }

    def predict_single_image_async(
        self,
        image_path: str,
        target_class_ids: List[int],
        threshold: float,
        poll_interval: float = 1.0,
        max_tries: int = 60,
    ) -> Dict:
        """Submits the job and polls for the result."""

        if not Path(image_path).exists():
            return {"error": f"image not found at {image_path}"}

        requests_json = PredictionRequest(
            image_path=image_path,
            target_class_ids=target_class_ids,
            threshold=threshold,
        )

        total_start_time = time.time()

        # 1. Submit the job (NON-BLOCKING on the server)
        try:
            response = self.session.post(
                f"{self.base_url}/predict", json=requests_json.dict()
            )
            response.raise_for_status()
            job_submission_result = response.json()

            if not job_submission_result.get("success"):
                return {"error": "Server failed to queue the job."}

            job_id = job_submission_result["job_id"]
            logger.info(f"Job submitted. ID: {job_id}")

        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to submit job to server: {e}"}

        # 2. Poll for the result (Client is now blocking, but the Server is not)
        result = self._poll_job_status(job_id, poll_interval, max_tries)

        total_request_time = (time.time() - total_start_time) * 1000
        logger.info(
            f"Total end-to-end request completed in {total_request_time:.2f} ms"
        )

        return result


if __name__ == "__main__":
    # Example usage
    client = MLInferenceClient(server_host="localhost", server_port=8000)
    status = client.check_server_status()
    logger.info(f"Server status: {status}")
    image_path = "/home/sky/Documents/job_applications/work_on_poetry/segmentation_api/segmentation_api/tests/images/cat_3.jpg"
    result = client.predict_single_image_async(
        image_path=image_path,
        target_class_ids=[8, 17],
        threshold=0.5,
    )
    logger.info(f"Prediction result: {result}")
