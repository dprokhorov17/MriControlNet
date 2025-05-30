from pydantic import BaseModel
from typing import Optional


class ProcessingRequest(BaseModel):
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
