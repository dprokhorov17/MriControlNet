"""Stable Diffusion and ControlNet pipeline management.

This module handles the initialization and management of the Stable Diffusion
pipeline with ControlNet integration, including model loading and device management.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import logging
import warnings
from config.settings import get_settings

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages the Stable Diffusion and ControlNet pipeline.

    Handles initialization, loading of models, and provides access to the
    pipeline for image generation. Implements memory optimization techniques
    like CPU offloading.
    """

    def __init__(self):
        self.pipeline = None
        self.settings = get_settings()

    def setup_pipeline(self):
        """Initialize and set up the Stable Diffusion pipeline with ControlNet.

        Loads the ControlNet and Stable Diffusion models, moves them to the
        appropriate device, and enables CPU offloading for memory optimization.

        Raises:
            Exception: If there's an error during pipeline setup
        """
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
        """Get the initialized pipeline instance.

        Returns:
            StableDiffusionControlNetPipeline: The initialized pipeline

        Raises:
            RuntimeError: If pipeline is not initialized
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline first.")
        return self.pipeline

    def is_loaded(self):
        """Check if the pipeline is initialized and ready.

        Returns:
            bool: True if pipeline is loaded, False otherwise
        """
        return self.pipeline is not None
