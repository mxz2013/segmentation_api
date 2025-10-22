from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from config.configure import VALID_MODEL_NAMES, DEFAULT_TARGETS, DEFAULT_THRESHOLD


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


class PredictionResponse(BaseModel):
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
    # device: str = Field(..., description="Device being used (cpu/cuda)")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """
    Error response schema
    """

    success: bool = Field(default=False, description="Success status")
    error_message: str = Field(..., description="Error description")
    error_code: str = Field(..., description="Error code")
