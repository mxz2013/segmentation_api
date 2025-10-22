import time
import threading
import sys
from typing import Dict
from pathlib import Path
import traceback
import logging
from fastapi import FastAPI, HTTPException
import uvicorn

from api.schemas import PredictionRequest, PredictionResponse, ServerStatus, Prediction

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from inference.segmenter import Segmenter

from config.configure import VALID_MODEL_NAMES

logger = logging.getLogger(__name__)


class MLInferenceServer:
    """
    FastAPI server for inference
    """

    def __init__(self, model_name: str, device: str):
        self.app = FastAPI(
            title="Segmentation API",
            description="FastAPI server for segmentation inference",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self.model_name = model_name
        self.device = device
        # model cache to avoid reloading
        self.model_cache: Dict[str, Segmenter] = {}
        # cache lock to protect model loading, as multiple requests can come at the same time
        # especially when using multiple workers, we need to ensure that only one thread loads the model at a time
        self.cache_lock = threading.Lock()
        self.setup_routes()

    def get_predictor(self) -> Segmenter:
        """
        get (if already loaded) or create a predictor for a model
        :return:
        """

        # 1. fast path: check the cache without a lock
        if self.model_name in self.model_cache:
            return self.model_cache[self.model_name]
        # 2. slow path: acquire lock to protect model loading
        # be sure that only one thread at a time can load the model
        with self.cache_lock:
            # 3 re-check: another thread might have finished loading while we waited for the lock
            if self.model_name not in self.model_cache:
                logger.info(f'Loading "{self.model_name}"')
                self.model_cache[self.model_name] = Segmenter(
                    model_name=self.model_name,
                    device=self.device,
                )
                logger.info(f"{self.model_name} loaded successfully")

        return self.model_cache[self.model_name]

    def setup_routes(self):
        """
        setup api routes
        :return:
        """

        @self.app.get("/", response_model=ServerStatus)
        async def root():
            return ServerStatus(
                status="running",
                available_models=list(VALID_MODEL_NAMES),
                version="1.0.0",
            )

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_single_image(request: PredictionRequest):
            start_time = time.time()

            try:
                if not Path(request.image_path).exists():
                    raise HTTPException(
                        status_code=404, detail=f"Image not found: {request.image_path}"
                    )

                predictor = self.get_predictor()

                predictions = predictor.segment(
                    request.image_path, request.target_class_ids, request.threshold
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

                return PredictionResponse(
                    image_path=request.image_path,
                    model_name=self.model_name,
                    predictions=predictions_objects,
                    processing_time_ms=processing_time,
                    success=True,
                )

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                error_msg = str(e)
                print(f"Prediction error: {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")

                return PredictionResponse(
                    image_path=request.image_path,
                    model_name=self.model_name,
                    predictions=[],
                    processing_time_ms=processing_time,
                    success=False,
                    error_message=error_msg,
                )

        @self.app.post("/preload_model")
        async def preload_model():
            """
            preload a model to cache
            """
            if self.model_name not in VALID_MODEL_NAMES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model name. Available {VALID_MODEL_NAMES}",
                )

            self.get_predictor()

            return {"success": True, "message": f"Model {self.model_name} preloaded"}


def run_server(
    model_name: str,
    device: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
):
    """
    :param model_name:
    :param device:
    :param host:
    :param port:
    :param workers:
    :return:
    """
    ml_server = MLInferenceServer(model_name=model_name, device=device)
    # force the model loading here to be sure it works for multiple workers
    try:
        ml_server.get_predictor()
        logger.info(f"Model {model_name} initialized during server startup.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model {model_name} during startup: {e}")

    app = ml_server.app
    uvicorn.run(
        # "api.server:app",
        app,
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info",
    )
