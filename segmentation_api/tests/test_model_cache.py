import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_caching():
    client = requests.Session()
    base_url = "http://localhost:8000"

    image_path = "/home/sky/Documents/job_applications/work_on_poetry/segmentation_api/segmentation_api/tests/images/cat_3.jpg"

    # Submit multiple jobs
    job_ids = []
    for i in range(5):
        response = client.post(
            f"{base_url}/predict",
            json={
                "image_path": image_path,
                "target_class_ids": [8, 17],
                "threshold": 0.5,
            },
        )
        job_data = response.json()
        job_ids.append(job_data["job_id"])
        logger.info(f"Submitted job {i+1}: {job_data['job_id']}")

    # Wait for results
    results = []
    for job_id in job_ids:
        while True:
            response = client.get(f"{base_url}/job_status/{job_id}")
            status_data = response.json()
            if status_data["status"] in ["finished", "failed"]:
                results.append(status_data)
                break
            time.sleep(0.5)

    # Analyze results
    finished = [r for r in results if r["status"] == "finished"]
    logger.info(f"✅ Completed: {len(finished)}/{len(job_ids)}")

    if finished:
        processing_times = [r["result"]["processing_time_ms"] for r in finished]
        logger.info(f"📊 Processing times: {[f'{t:.1f}ms' for t in processing_times]}")
        logger.info(f"📈 First job: {processing_times[0]:.1f}ms")
        logger.info(f"📈 Subsequent jobs: {processing_times[1:]}")


if __name__ == "__main__":
    test_model_caching()
