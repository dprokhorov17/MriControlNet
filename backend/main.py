"""FastAPI backend for MRI ControlNet image processing.

This module provides the REST API endpoints for the MRI ControlNet application,
handling image uploads, processing requests, and serving processed results.
The API supports image processing with ControlNet and color transfer operations.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64
import uvicorn
import json
import logging

# Internal imports
from config.settings import get_settings
from data_models.processing_request import ProcessingRequest
from pipeline_manager import PipelineManager
from controlnet_handler import ControlNetHandler
from color_transfer import ColorTransfer
from image_processor import ImageProcessor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# Initialize components
app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)
pipeline_manager = PipelineManager()
controlnet_handler = ControlNetHandler(pipeline_manager)
color_transfer = ColorTransfer()
image_processor = ImageProcessor()


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 string.

    Args:
        image (Image.Image): PIL Image to convert

    Returns:
        str: Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup.

    Sets up the ControlNet and Stable Diffusion pipeline.
    """
    try:
        logger.info("Initializing pipeline on startup...")
        pipeline_manager.setup_pipeline()
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")


@app.post("/process")
async def process_image(file: UploadFile = File(...), params: str = ""):
    """Process an uploaded image using ControlNet and color transfer.

    Args:
        file (UploadFile): The input image file
        params (str): JSON string containing processing parameters

    Returns:
        JSONResponse: Dictionary containing base64-encoded versions of:
            - original: Input image
            - control: Edge detection result
            - generated: ControlNet generation
            - color_transferred: Final processed image

    Raises:
        HTTPException: If processing fails
    """
    try:
        if params:
            params_dict = json.loads(params)
            processing_params = ProcessingRequest(**params_dict)
        else:
            processing_params = ProcessingRequest(prompt="high quality image")

        contents = await file.read()
        input_image = Image.open(BytesIO(contents))
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        processed = image_processor.process(
            input_image, processing_params, controlnet_handler, color_transfer
        )

        results = {
            "original": image_to_base64(input_image),
            "control": image_to_base64(processed["control_image"]),
            "generated": image_to_base64(processed["generated_image"]),
            "color_transferred": image_to_base64(processed["color_transferred"]),
        }

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of the application.

    Returns:
        dict: Health status and pipeline state
    """
    return {"status": "healthy", "pipeline_loaded": pipeline_manager.is_loaded()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
