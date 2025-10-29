from pathlib import Path

import logging
from fastapi import FastAPI, HTTPException
import uvicorn
import redis

from rq import Queue
from rq.job import Job

from rq.exceptions import NoSuchJobError
from segmentation_api.config.configure import VALID_MODEL_NAMES


from segmentation_api.api.worker import perform_segmentation_task


from segmentation_api.api.schemas import (
    PredictionRequest,
    WorkerPredictionResponse,
    ServerStatus,
    ServerTaskData,
    JobStatusResponse,
    ServerQueueStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_conn = redis.Redis(host="localhost", port=6379)
inference_queue = Queue("ml_inference_queue", connection=redis_conn)


class MLInferenceServer:
    """
    The ML inference server using FastAPI and RQ for task queueing.
    """

    def __init__(self, model_name: str, device: str):
        self.app = FastAPI(
            title="ML Inference Server",
            description="A server for ML model inference using FastAPI and RQ",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self.model_name = model_name
        self.device = device
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/", response_model=ServerStatus)
        async def root():
            """
            Get server status.
            """
            return ServerStatus(
                status="running",
                available_models=list(VALID_MODEL_NAMES),
                version="1.0.0",
            )

        # the /predict endpoint submites a job and returns an immediate response with job id
        @self.app.post("/predict", response_model=ServerQueueStatus)
        async def predict_single_image(request: PredictionRequest):
            """
            Submit a prediction request for a single image.
            """
            if not Path(request.image_path).exists():
                raise HTTPException(
                    status_code=404, detail=f"Image not found at {request.image_path}"
                )

            task_data = ServerTaskData(
                image_path=request.image_path,
                target_class_ids=request.target_class_ids,
                threshold=request.threshold,
                model_name=self.model_name,
                device=self.device,
            )

            try:
                job: Job = inference_queue.enqueue(
                    perform_segmentation_task,
                    task_data.dict(),  # Convert to dict for serialization
                )
                logger.info(f"Enqueued job {job.id} for image {request.image_path}")

                # Return a response indicating the job has been accepted
                return ServerQueueStatus(
                    success=True,
                    job_id=job.id,
                    status=f"/job_status/{job.id}",
                    position_in_queue=job.get_position(),
                )

            except Exception as e:
                logger.error(
                    f"Failed to enqueue job for image {request.image_path}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500, detail="Failed to enqueue prediction job."
                )

        # New endpoint to check job status and get results
        @self.app.get("/job_status/{job_id}", response_model=JobStatusResponse)
        async def get_job_status(job_id: str):
            """
            Get the status and result of a prediction job.
            """
            try:
                job: Job = Job.fetch(job_id, connection=redis_conn)
                if job.is_finished:
                    result = job.result

                    return JobStatusResponse(
                        job_id=job_id,
                        status="finished",
                        result=result,
                    )
                elif job.is_failed:
                    # instead of racing an exception, return the error in the response
                    # Create error response
                    result = job.result
                    error_result = WorkerPredictionResponse(
                        image_path=result.image_path if result else "",
                        model_name=result.model_name if result else "",
                        predictions=[],
                        processing_time_ms=0,
                        success=False,
                        error_message=result.get("error_message", "Job failed."),
                    )
                    return JobStatusResponse(
                        job_id=job_id,
                        status="failed",
                        result=error_result,
                    )
                else:  # job is still in progress
                    return JobStatusResponse(
                        job_id=job_id,
                        status=job.get_status(),
                        result=None,
                    )
            except NoSuchJobError:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            except Exception as e:
                logger.error(f"Error fetching job {job_id}: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )

        # Preload is now redundant in the server, but keep the route for consistency
        @self.app.post("/preload_model")
        async def preload_model():
            return {
                "success": True,
                "message": "Model preload managed by worker process.",
            }


def run_server(
    model_name: str,
    device: str,
    host: str = "localhost",
    port: int = 8000,
    workers: int = 1,  # Uvicorn workers still here for API concurrency
):
    ml_server = MLInferenceServer(model_name=model_name, device=device)
    app = ml_server.app
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info",
    )


if __name__ == "__main__":
    run_server(
        model_name="google/deeplabv3_mobilenet_v2_1.0_513",
        device="cpu",
        host="localhost",
        port=8000,
        workers=1,
    )
