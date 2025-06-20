"""ControlNet image processing handler.

This module manages the interaction with the ControlNet model for image processing,
including edge detection and image generation using Stable Diffusion with ControlNet guidance.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import logging

from config.settings import get_settings
from data_models.processing_request import ProcessingRequest
from pipeline_manager import PipelineManager


logger = logging.getLogger(__name__)


class ControlNetHandler:
    """Handles ControlNet-based image processing operations.

    This class manages the interaction with ControlNet models, including
    edge detection preprocessing and controlled image generation.

    Args:
        pipeline_manager (PipelineManager): Manager for the Stable Diffusion pipeline
    """

    def __init__(self, pipeline_manager: PipelineManager):
        self.pipeline_manager = pipeline_manager
        self.settings = get_settings()

    def apply_canny(self, image, low_threshold=100, high_threshold=200):
        """Apply Canny edge detection to the input image.

        Args:
            image: Input image (RGB or grayscale)
            low_threshold (int): Lower threshold for edge detection
            high_threshold (int): Upper threshold for edge detection

        Returns:
            PIL.Image: Edge detection result as RGB image
        """

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        canny = cv2.Canny(image, low_threshold, high_threshold)
        canny_image = np.stack([canny] * 3, axis=2)
        return Image.fromarray(canny_image)

    def process_with_controlnet(self, input_image, params: ProcessingRequest):
        """Process an image using ControlNet-guided Stable Diffusion.

        Args:
            input_image (Union[np.ndarray, PIL.Image]): Input image to process
            params (ProcessingRequest): Processing parameters

        Returns:
            PIL.Image: Generated image guided by ControlNet
        """

        pipeline = self.pipeline_manager.get_pipeline()

        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        input_image = input_image.resize(
            (params.image_resolution, params.image_resolution)
        )

        control_image = self.apply_canny(
            np.array(input_image), params.low_threshold, params.high_threshold
        )

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.settings.DEVICE).manual_seed(
                params.seed
            )

        result = pipeline(
            prompt=params.prompt,
            image=control_image,
            negative_prompt=params.negative_prompt,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            controlnet_conditioning_scale=params.controlnet_conditioning_scale,
            generator=generator,
            return_dict=False,
        )

        generated_image = result[0][0] if isinstance(result[0], list) else result[0]

        return control_image, generated_image
