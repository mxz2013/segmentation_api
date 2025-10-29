import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConcurrentTestClient:
    def __init__(self, server_host: str = "localhost", server_port: int = 8000):
        self.base_url = f"http://{server_host}:{server_port}"
        self.session = requests.Session()

    def submit_single_job(
        self, job_id: int, image_path: str, target_class_ids: list, threshold: float
    ):
        """Submit a single job and track its progress"""
        start_time = time.time()
        logger.info(f"🚀 Job {job_id}: Submitting at {time.strftime('%H:%M:%S')}")

        try:
            # Submit job
            response = self.session.post(
                f"{self.base_url}/predict",
                json={
                    "image_path": image_path,
                    "target_class_ids": target_class_ids,
                    "threshold": threshold,
                },
            )
            response.raise_for_status()
            job_data = response.json()

            submission_time = time.time() - start_time
            logger.info(
                f"✅ Job {job_id}: Submitted in {submission_time:.2f}s - Job ID: {job_data['job_id']} - Position: {job_data.get('position_in_queue', 'N/A')}"
            )

            # Poll for result
            job_start_time = time.time()
            final_result = self._poll_job_result(job_data["job_id"], job_id)
            job_total_time = time.time() - job_start_time

            if final_result.get("status") == "finished":
                logger.info(
                    f"🎉 Job {job_id}: COMPLETED in {job_total_time:.2f}s - Predictions: {len(final_result['result']['predictions'])}"
                )
            elif final_result.get("status") == "failed":
                logger.error(
                    f"💥 Job {job_id}: FAILED in {job_total_time:.2f}s - Error: {final_result['result']['error_message']}"
                )
            else:
                logger.warning(
                    f"⚠️ Job {job_id}: UNKNOWN STATUS after {job_total_time:.2f}s"
                )

            return {
                "job_id": job_id,
                "submission_time": submission_time,
                "total_time": job_total_time,
                "status": final_result.get("status"),
                "result": final_result,
            }

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"💥 Job {job_id}: ERROR in {error_time:.2f}s - {str(e)}")
            return {"job_id": job_id, "error": str(e), "submission_time": error_time}

    def _poll_job_result(self, job_id: str, original_job_id: int):
        """Poll job status until completion"""
        poll_count = 0
        while poll_count < 120:  # Max 2 minutes
            try:
                response = self.session.get(f"{self.base_url}/job_status/{job_id}")
                response.raise_for_status()
                status_data = response.json()

                if status_data["status"] in ["finished", "failed"]:
                    return status_data

                poll_count += 1
                if poll_count % 10 == 0:  # Log every 10 polls
                    logger.info(
                        f"⏳ Job {original_job_id}: Still processing... (poll {poll_count})"
                    )

                time.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"❌ Job {original_job_id}: Polling error - {str(e)}")
                break

        return {"status": "timeout", "error": "Polling timeout"}


def run_concurrent_test(num_requests=10):
    """Run concurrent test with multiple requests"""
    client = ConcurrentTestClient()

    # Test image path - make sure this exists
    image_path = "/home/sky/Documents/job_applications/work_on_poetry/segmentation_api/segmentation_api/tests/images/cat_3.jpg"

    if not Path(image_path).exists():
        logger.error(f"❌ Test image not found: {image_path}")
        return

    logger.info(f"🎬 Starting concurrent test with {num_requests} requests")
    logger.info(f"📊 Target image: {image_path}")
    logger.info(f"⏰ Start time: {time.strftime('%H:%M:%S')}")

    start_time = time.time()

    # Submit all jobs concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(
                client.submit_single_job,
                job_id=i + 1,
                image_path=image_path,
                target_class_ids=[8, 17],
                threshold=0.5,
            )
            futures.append(future)

        # Wait for all jobs to complete and collect results
        results = [future.result() for future in futures]

    total_test_time = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 TEST SUMMARY")
    logger.info("=" * 60)

    completed = [r for r in results if r.get("status") == "finished"]
    failed = [r for r in results if r.get("status") == "failed"]
    errors = [r for r in results if r.get("error")]

    logger.info(f"✅ Completed: {len(completed)}")
    logger.info(f"❌ Failed: {len(failed)}")
    logger.info(f"💥 Errors: {len(errors)}")
    logger.info(f"⏱️ Total test time: {total_test_time:.2f}s")

    if completed:
        total_times = [r["total_time"] for r in completed]
        avg_time = sum(total_times) / len(total_times)
        max_time = max(total_times)
        min_time = min(total_times)

        logger.info(f"📊 Average processing time: {avg_time:.2f}s")
        logger.info(f"📊 Fastest job: {min_time:.2f}s")
        logger.info(f"📊 Slowest job: {max_time:.2f}s")
        logger.info(
            f"📊 Total queue throughput: {len(completed)/total_test_time:.2f} jobs/second"
        )

    # Print individual job results
    logger.info("\n" + "=" * 60)
    logger.info("📋 INDIVIDUAL JOB RESULTS")
    logger.info("=" * 60)

    for result in sorted(results, key=lambda x: x["job_id"]):
        if "error" in result:
            logger.info(f"Job {result['job_id']}: ❌ ERROR - {result['error']}")
        else:
            status_icon = "✅" if result["status"] == "finished" else "❌"
            logger.info(
                f"Job {result['job_id']}: {status_icon} {result['status'].upper()} - {result['total_time']:.2f}s"
            )


if __name__ == "__main__":
    run_concurrent_test(num_requests=10)
