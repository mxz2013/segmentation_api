import requests
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from segmentation_api.api.client import MLInferenceClient
from segmentation_api.config.configure import DEFAULT_TARGETS, DEFAULT_THRESHOLD


# Configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8000
NUM_CONCURRENT_USERS = 10
IMAGE_PATH_FOR_TEST = "tests/images/cat_3.jpg"


def compute_performance_statistics(results: List[Dict], total_duration: float) -> Dict:
    """
    compute performance statistics of the api under concurrent calls
    Args:
        results: The results of the requests
        total_duration: the total duration of the test in seconds

    Returns:

    """
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    stats = {
        "total_requests": NUM_CONCURRENT_USERS,
        "successful_requests": len(successful_results),
        "failed_requests": len(failed_results),
        "success_rate": (
            (len(successful_results) / len(results) * 100) if results else 0
        ),
        "failed_rate": (len(failed_results) / len(results) * 100) if results else 0,
    }

    if successful_results:
        times = [r["time_ms"] for r in successful_results]

        stats["response_time"] = {
            "mean": np.mean(times),
            "median": np.median(times),
            "min": np.min(times),
            "max": np.max(times),
            "std_dev": np.std(times),
        }
        # Percentiles (critical for understanding tail latency)
        stats["percentiles"] = {
            "p50": np.percentile(times, 50),
            "p75": np.percentile(times, 75),
            "p90": np.percentile(times, 90),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
        }

        # Throughput metrics
        stats["throughput"] = {
            "requests_per_second": len(successful_results) / total_duration,
            "total_duration_seconds": total_duration,
            "avg_requests_per_second_per_request": 1000
            / stats["response_time"]["mean"],
        }

        # Concurrency efficiency
        # Ideal concurrent time = max(individual times)
        # Actual time = total_duration
        stats["concurrency"] = {
            "total_wall_clock_time_s": total_duration,
            "longest_individual_request_ms": max(times),
            "concurrency_efficiency": (
                (max(times) / 1000) / total_duration if total_duration > 0 else 0
            ),
        }

    return stats


def print_performance_report(stats: Dict):
    """
    Print a formatted performance report.
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)

    print(f"\n Overall Metrics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(
        f"  Successful: {stats['successful_requests']} ({stats['success_rate']:.1f}%)"
    )
    print(f"  Failed: {stats['failed_requests']} ({stats['failed_rate']:.1f}%)")

    if "response_time" in stats:
        print(f"\n Response Time Statistics:")
        rt = stats["response_time"]
        print(f"  Mean:    {rt['mean']:.2f} ms")
        print(f"  Median:  {rt['median']:.2f} ms")
        print(f"  Std Dev: {rt['std_dev']:.2f} ms")
        print(f"  Min:     {rt['min']:.2f} ms")
        print(f"  Max:     {rt['max']:.2f} ms")

        print(f"\n Percentiles (Tail Latency):")
        p = stats["percentiles"]
        print(f"  P50 (Median): {p['p50']:.2f} ms")
        print(f"  P75:          {p['p75']:.2f} ms")
        print(f"  P90:          {p['p90']:.2f} ms")
        print(f"  P95:          {p['p95']:.2f} ms")
        print(f"  P99:          {p['p99']:.2f} ms")

        print(f"\nThroughput:")
        th = stats["throughput"]
        print(f"  Requests/Second: {th['requests_per_second']:.2f} req/s")
        print(f"  Total Duration:  {th['total_duration_seconds']:.3f} s")

        print(f"\n Concurrency:")
        conc = stats["concurrency"]
        print(f"  Wall Clock Time:      {conc['total_wall_clock_time_s']:.3f} s")
        print(f"  Longest Request:      {conc['longest_individual_request_ms']:.2f} ms")
        print(f"  Efficiency:           {conc['concurrency_efficiency'] * 100:.1f}%")

    print("\n" + "=" * 60)


def run_single_user_prediction(user_id: int) -> Dict:
    """
    Simulates a single user making a prediction request to the API.
    Args:
        user_id: The identifier for the simulated user.

    Returns: A dictionary with the user_id, response time, and success status.
    """

    client = MLInferenceClient(server_host=SERVER_HOST, server_port=SERVER_PORT)
    start_time = time.time()

    try:
        # Call the synchronous prediction method
        result = client.predict_single_image(
            image_path=IMAGE_PATH_FOR_TEST,
            target_class_ids=DEFAULT_TARGETS,
            threshold=DEFAULT_THRESHOLD,
        )
        response_time = (time.time() - start_time) * 1000  # in ms

        # Basic validation (Success and no client-side error)
        if "error" in result:
            raise Exception(f"Client-side image path error: {result['error']}")

        # Server-side validation
        if not result.get("success"):
            error_message = result.get("error_message", "Unknown server error")
            raise Exception(f"Server-side prediction failed: {error_message}")
        print(f"User {user_id}: Success. Response Time: {response_time:.2f}ms")
        return {"user_id": user_id, "time_ms": response_time, "success": True}

    except requests.exceptions.RequestException as e:
        print(f"User {user_id}: Connection/HTTP error: {e}")
        return {
            "user_id": user_id,
            "success": False,
            "error": f"Connection/HTTP Error: {e}",
        }
    except Exception as e:
        print(f"User {user_id}: Prediction failed: {e}")
        return {"user_id": user_id, "success": False, "error": str(e)}


def test_concurrent_predictions_sync() -> None:
    """
    Simulates N concurrent users calling the API using ThreadPoolExecutor.
    Returns:
    """

    print(f"\n--- Starting {NUM_CONCURRENT_USERS} concurrent requests ---")

    results = []

    client = MLInferenceClient(SERVER_HOST, SERVER_PORT)
    print("Preloading model via /preload_model endpoint...")
    # Preload the model before starting the test to avoid loading time affecting results
    client.session.post(f"{client.base_url}/preload_model").raise_for_status()
    print("Model preloaded successfully.")
    test_start_time = time.time()

    # Initialize the ThreadPoolExecutor with the number of concurrent users
    # API requests are I/O-bound, so threads are appropriate here
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_USERS) as executor:
        # Submit the prediction function for each user
        future_to_user = {
            executor.submit(run_single_user_prediction, i): i
            for i in range(1, NUM_CONCURRENT_USERS + 1)
        }

        for future in as_completed(future_to_user):
            try:
                result = future.result()  # Get the return value from the thread
                results.append(result)
            except Exception as exc:
                print(f"User {future_to_user[future]} generated an exception: {exc}")
                # Add a failure result to maintain count
                results.append(
                    {
                        "user_id": future_to_user[future],
                        "success": False,
                        "error": str(exc),
                    }
                )

    total_duration = time.time() - test_start_time
    print("\n--- All requests completed ---")
    stats = compute_performance_statistics(results, total_duration)
    print_performance_report(stats)


if __name__ == "__main__":
    test_concurrent_predictions_sync()
