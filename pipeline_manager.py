import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import logging
import warnings
from config.settings import get_settings

logger = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self):
        self.pipeline = None
        self.settings = get_settings()

    def setup_pipeline(self):
        try:
            logger.info("Loading ControlNet model...")
            controlnet = ControlNetModel.from_pretrained(
                self.settings.DEFAULT_CONTROLNET_MODEL, torch_dtype=torch.float16
            )

            logger.info("Loading Stable Diffusion model...")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.settings.DEFAULT_SD_MODEL,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = pipe.to(self.settings.DEVICE)

            pipe.enable_model_cpu_offload()
            self.pipeline = pipe
            logger.info("Pipeline loaded successfully!")
        except Exception as e:
            logger.error(f"Error setting up pipeline: {e}")
            raise

    def get_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline first.")
        return self.pipeline

    def is_loaded(self):
        return self.pipeline is not None
