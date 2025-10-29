from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from segmentation_api.config.configure import DEFAULT_TARGETS, DEFAULT_THRESHOLD


class PredictionRequest(BaseModel):
    """
    request schema for single image inference
    TODO: for batch inference
    """

    image_path: str = Field(..., description="Path to the image file")
    threshold: float = Field(
        default=DEFAULT_THRESHOLD, description="Threshold for binary mask"
    )
    target_class_ids: List[int] = Field(
        default=DEFAULT_TARGETS, description="targe class ids"
    )

    @field_validator("image_path")
    def validate_image_path(cls, v: str):
        if not v.strip():
            raise ValueError("Image path cannot be empty")
        return v.strip()


class Prediction(BaseModel):
    """
    schema for inference results
    we only save scores, and pix_count
    """

    class_id: int = Field(..., description="The target class id")
    pix_count: int = Field(
        ..., description="The number of pixels for the detected mask"
    )


class WorkerPredictionResponse(BaseModel):
    """
    schema for prediction response
    """

    image_path: str = Field(..., description="Path to the processed image")
    model_name: str = Field(..., description="Model used for inference")
    predictions: List[Prediction] = Field(..., description="List of predictions")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    success: bool = Field(default=True, description="Whether prediction was successful")
    error_message: Optional[str] = Field(
        default=None, description="Error message if any"
    )


class ServerStatus(BaseModel):
    """
    Server status response
    """

    status: str = Field(..., description="Server status")
    available_models: List[str] = Field(..., description="List of available models")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """
    Error response schema
    """

    success: bool = Field(default=False, description="Success status")
    error_message: str = Field(..., description="Error description")
    error_code: str = Field(..., description="Error code")


class ServerTaskData(BaseModel):
    """
    Schema for task data sent to the worker
    """

    image_path: str = Field(..., description="Path to the image file")
    threshold: float = Field(
        default=DEFAULT_THRESHOLD, description="Threshold for binary mask"
    )
    target_class_ids: List[int] = Field(
        default=DEFAULT_TARGETS, description="Target class ids"
    )
    model_name: Optional[str] = Field(
        default=None, description="Model name to use for inference"
    )
    device: Optional[str] = Field(
        default=None, description="Device to run the model on (e.g., 'cpu', 'cuda')"
    )


class JobStatusResponse(BaseModel):
    """
    Schema for job status response
    """

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    result: Optional[WorkerPredictionResponse] = Field(
        default=None, description="Prediction result if job is completed"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )


class ServerQueueStatus(BaseModel):
    """
    Schema for server queue status
    """

    job_id: str = Field(..., description="Job identifier")
    success: bool = Field(
        default=True, description="Whether the operation was successful"
    )
    status: str = Field(..., description="Current status of the job")
    position_in_queue: Optional[int] = Field(
        default=None, description="Position in the queue"
    )


class ClientRequest(BaseModel):
    """
    Client request schema for submitting a prediction job
    """

    image_path: str = Field(..., description="Path to the image file")
    threshold: float = Field(
        default=DEFAULT_THRESHOLD, description="Threshold for binary mask"
    )
    target_class_ids: List[int] = Field(
        default=DEFAULT_TARGETS, description="Target class ids"
    )
