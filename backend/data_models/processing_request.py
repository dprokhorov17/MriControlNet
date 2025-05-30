"""Data model for image processing requests.

This module defines the data structure for image processing parameters used in the MRI ControlNet pipeline.
"""
from pydantic import BaseModel
from typing import Optional


class ProcessingRequest(BaseModel):
    """Configuration parameters for image processing using ControlNet and Stable Diffusion.

    Attributes:
        prompt (str): The text prompt for Stable Diffusion image generation
        negative_prompt (str): Text prompt for features to avoid in generation
        num_inference_steps (int): Number of denoising steps (default: 20)
        guidance_scale (float): How closely to follow the prompt (default: 7.5)
        controlnet_conditioning_scale (float): Strength of ControlNet conditioning (default: 1.0)
        low_threshold (int): Lower threshold for edge detection (default: 100)
        high_threshold (int): Upper threshold for edge detection (default: 200)
        seed (Optional[int]): Random seed for reproducible generation
        image_resolution (int): Output image size in pixels (default: 512)
        color_transfer_mode (str): Color transfer algorithm to use (default: "lab")
        color_transfer_strength (float): Intensity of color transfer (default: 1.0)
    """

    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    low_threshold: int = 100
    high_threshold: int = 200
    seed: Optional[int] = None
    image_resolution: int = 512
    color_transfer_mode: str = "lab"  # "lab", "yuv", or "luminance"
    color_transfer_strength: float = 1.0
