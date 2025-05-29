import numpy as np
import cv2
from PIL import Image
import torch
import logging

from config.settings import get_settings
from models.processing_request import ProcessingRequest
from pipeline_manager import PipelineManager


logger = logging.getLogger(__name__)


class ControlNetHandler:
    def __init__(self, pipeline_manager: PipelineManager):
        self.pipeline_manager = pipeline_manager
        self.settings = get_settings()

    def apply_canny(self, image, low_threshold=100, high_threshold=200):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        canny = cv2.Canny(image, low_threshold, high_threshold)
        canny_image = np.stack([canny] * 3, axis=2)
        return Image.fromarray(canny_image)

    def process_with_controlnet(self, input_image, params: ProcessingRequest):
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
