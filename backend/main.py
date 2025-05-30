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
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing pipeline on startup...")
        pipeline_manager.setup_pipeline()
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")


@app.post("/process")
async def process_image(file: UploadFile = File(...), params: str = ""):
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
    return {"status": "healthy", "pipeline_loaded": pipeline_manager.is_loaded()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
