"""Main image processing orchestration module.

This module coordinates the image processing workflow, including ControlNet
processing and color transfer operations.
"""

import numpy as np
import logging
from PIL import Image

from color_transfer import ColorTransfer
from controlnet_handler import ControlNetHandler
from data_models.processing_request import ProcessingRequest


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Orchestrates the complete image processing workflow.

    This class coordinates between ControlNet processing and color transfer
    operations to generate and refine images based on input parameters.
    """

    def process(
        self,
        input_image: Image.Image,
        processing_params: ProcessingRequest,
        controlnet_handler: ControlNetHandler,
        color_transfer: ColorTransfer,
    ):
        """Process an input image using ControlNet and color transfer.

        Args:
            input_image (Image.Image): The source image to process
            processing_params (ProcessingRequest): Processing configuration
            controlnet_handler (ControlNetHandler): Handler for ControlNet operations
            color_transfer (ColorTransfer): Handler for color transfer operations

        Returns:
            dict: Dictionary containing:
                - control_image: Edge detection result
                - generated_image: ControlNet generation result
                - color_transferred: Final image with color transfer applied

        Raises:
            Exception: If processing fails at any stage
        """
        try:
            control_image, generated_image = controlnet_handler.process_with_controlnet(
                input_image, processing_params
            )

            input_resized = np.array(
                input_image.resize(
                    (
                        processing_params.image_resolution,
                        processing_params.image_resolution,
                    )
                )
            )

            generated_array = np.array(generated_image)

            color_transferred = (
                color_transfer.take_luminance_from_first_chroma_from_second(
                    input_resized,
                    generated_array,
                    mode=processing_params.color_transfer_mode,
                    s=processing_params.color_transfer_strength,
                )
            )

            return {
                "control_image": control_image,
                "generated_image": generated_image,
                "color_transferred": Image.fromarray(
                    color_transferred.astype(np.uint8)
                ),
            }

        except Exception as e:
            logger.error(f"Error in image processing pipeline: {e}")
            raise
