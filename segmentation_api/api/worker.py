# This file defines a worker process that listens to an RQ queue for segmentation tasks.
# It loads the heavy Segmenter module and processes tasks synchronously.
import os
import time
import redis
from rq import Worker, SimpleWorker
import logging
from typing import Dict, Any
import threading

from segmentation_api.inference.segmenter import Segmenter
from segmentation_api.config.configure import VALID_MODEL_NAMES
from segmentation_api.api.schemas import Prediction, WorkerPredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_conn = redis.Redis(host="localhost", port=6379)

# Global cache for models in this worker process
_model_cache = {}
_cache_lock = threading.Lock()


def get_or_load_model(model_name: str, device: str) -> Segmenter:
    """
    Get the Segmenter model from cache or load it if not present.
    This uses global process-level storage that persists across jobs.
    """
    global _model_cache, _cache_lock

    # Create cache key that includes both model_name AND device
    cache_key = f"{model_name}||{device}"

    logger.info(f"🔍 PID {os.getpid()} checking cache for: '{cache_key}'")
    logger.info(
        f"📦 Current cache keys in PID {os.getpid()}: {list(_model_cache.keys())}"
    )

    # 1. Fast path: check cache without lock
    if cache_key in _model_cache:
        logger.info(f'✅ CACHE HIT in PID {os.getpid()}: "{cache_key}"')
        return _model_cache[cache_key]
    else:
        logger.info(f'❌ CACHE MISS in PID {os.getpid()}: "{cache_key}"')

    # 2. Slow path: acquire lock to load model
    with _cache_lock:
        # Double-check after acquiring lock
        if cache_key not in _model_cache:
            logger.info(
                f'🚀 LOADING in PID {os.getpid()}: "{model_name}" on device "{device}"'
            )
            start_time = time.time()

            # This will handle both CPU and GPU loading automatically
            _model_cache[cache_key] = Segmenter(
                model_name=model_name,
                device=device,
            )

            load_time = time.time() - start_time
            logger.info(
                f"✅ LOADED in PID {os.getpid()}: {cache_key} in {load_time:.2f}s"
            )
            logger.info(
                f"📦 Cache in PID {os.getpid()} now has: {list(_model_cache.keys())}"
            )
        else:
            logger.info(f'♻️ Another thread loaded "{cache_key}" while we waited')

    return _model_cache[cache_key]


def perform_segmentation_task(task_data: Dict[str, Any]) -> WorkerPredictionResponse:
    """
    Perform segmentation task using Segmenter.
    Works for both CPU and GPU inference.

    Args:
        task_data: Dictionary containing 'image_path', 'target_class_ids', 'threshold', and 'model_name'.

    Returns:
        Dictionary with prediction results.
    """
    image_path = task_data["image_path"]
    target_class_ids = task_data["target_class_ids"]
    threshold = task_data["threshold"]
    model_name = task_data.get("model_name", "google/deeplabv3_mobilenet_v2_1.0_513")
    device = task_data.get("device", "cpu")

    worker_pid = os.getpid()
    logger.info(f"🔧 Worker PID {worker_pid}: Starting job for image {image_path}")
    logger.info(
        f"📦 Job details - Model: {model_name}, Device: {device}, Target classes: {target_class_ids}"
    )

    if model_name not in VALID_MODEL_NAMES:
        raise ValueError(f"Invalid model name: {model_name}")

    logger.info(f"Loading model {model_name} on device {device}")
    start_time = time.time()
    try:
        # This will load the model on the correct device (CPU/GPU)
        # and cache it separately for each device
        segmenter = get_or_load_model(model_name=model_name, device=device)

        logger.info(f"Performing segmentation on image: {image_path}")
        predictions = segmenter.segment(
            image_path=image_path,
            target_class_ids=target_class_ids,
            threshold=threshold,
        )

        # convert predictions to response format
        predictions_objects = [
            Prediction(
                class_id=class_id,
                pix_count=pix_count,
            )
            for class_id, pix_count in predictions.items()
        ]
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"{predictions_objects = }")

        # Return the structured results (RQ handles storing this in Redis)
        return WorkerPredictionResponse(
            image_path=image_path,
            model_name=model_name,
            predictions=predictions_objects,
            processing_time_ms=processing_time,
            success=True,
            error_message=None,
        )

    except Exception as e:
        # Handle errors and store them in the result for the client to retrieve
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return WorkerPredictionResponse(
            image_path=image_path,
            model_name=model_name,
            predictions=[],
            processing_time_ms=processing_time,
            success=False,
            error_message=str(e),
        )


class PreloadingWorker(SimpleWorker):
    """
    Custom worker that inherits from SimpleWorker (no forking).
    This ensures models loaded into memory persist across jobs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preload models when worker instance is created
        logger.info("🚀 Preloading models in worker process...")
        logger.info(f"📍 Worker PID: {os.getpid()}")
        # Make sure this matches the device you'll be using in production
        get_or_load_model("google/deeplabv3_mobilenet_v2_1.0_513", "cpu")
        logger.info("✅ Worker startup complete - models loaded and cached")


if __name__ == "__main__":
    listen = ["ml_inference_queue"]

    # CRITICAL: Disable forking to maintain model cache across jobs
    worker = PreloadingWorker(
        listen,
        connection=redis_conn,
    )

    # Work WITHOUT forking - this keeps the cache alive
    logger.info("🎯 Starting worker WITHOUT forking (cache will persist)")
    worker.work(with_scheduler=False, burst=False)

    # Work in the same process - cache persists across all jobs
    logger.info("🎯 Starting SimpleWorker (no forking - cache will persist)")
    logger.info(f"📍 Main worker PID: {os.getpid()}")
    worker.work(with_scheduler=False)
