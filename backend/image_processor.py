import numpy as np
import logging
from PIL import Image

from color_transfer import ColorTransfer
from controlnet_handler import ControlNetHandler
from data_models.processing_request import ProcessingRequest


logger = logging.getLogger(__name__)


class ImageProcessor:

    def process(
        self,
        input_image: Image.Image,
        processing_params: ProcessingRequest,
        controlnet_handler: ControlNetHandler,
        color_transfer: ColorTransfer,
    ):
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
