import gradio as gr
from PIL import Image
import base64
import requests
from io import BytesIO
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/process")


def process_image_with_controlnet(
    image,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    controlnet_conditioning_scale,
    low_threshold,
    high_threshold,
    seed,
    image_resolution,
    color_transfer_mode,
    color_transfer_strength,
):
    """
    Sends the uploaded image and parameters to FastAPI backend and returns processed images.
    """
    if image is None:
        raise ValueError("No image provided")

    # Convert PIL image to base64 string
    buffered = BytesIO()
    image.save(buffered, format=image.format if image.format else "PNG")
    img_bytes = buffered.getvalue()  # Raw image bytes

    # Prepare JSON payload for params
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "low_threshold": int(low_threshold),
        "high_threshold": int(high_threshold),
        "seed": int(seed) if seed != -1 else None,
        "image_resolution": int(image_resolution),
        "color_transfer_mode": color_transfer_mode,
        "color_transfer_strength": float(color_transfer_strength),
    }

    # Prepare form data
    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"params": json.dumps(payload)}

    # Send POST request to FastAPI backend
    response = requests.post(API_URL, data=data, files=files)

    if response.status_code != 200:
        raise Exception(f"Error from API: {response.text}")

    result = response.json()

    # Decode base64 strings back to PIL Images
    def decode_image(b64_str):
        return Image.open(BytesIO(base64.b64decode(b64_str)))

    control = decode_image(result["control"])
    generated = decode_image(result["generated"])
    color_transferred = decode_image(result["color_transferred"])

    return color_transferred, control, generated


# Gradio Interface UI Components

title = "🎨 MRI ControlNet Image Processing UI"
description = """
Upload an brain scan image and adjust parameters to generate synthetic bran scan images using ControlNet.
"""

inputs = [
    gr.Image(type="pil", label="📤 Upload Image (PNG or JPG)", height=300),
    gr.Textbox(label="✍️ Prompt", value="High quality image", lines=2),
    gr.Textbox(label="🚫 Negative Prompt", value="", lines=2),
    gr.Slider(minimum=10, maximum=100, step=1, value=20, label="🔢 Inference Steps"),
    gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=7.5, label="⚖️ Guidance Scale"),
    gr.Slider(
        minimum=0.1,
        maximum=2.0,
        step=0.1,
        value=1.0,
        label="🔗 ControlNet Conditioning Scale",
    ),
    gr.Slider(
        minimum=1, maximum=255, step=1, value=100, label="📉 Canny Low Threshold"
    ),
    gr.Slider(
        minimum=1, maximum=255, step=1, value=200, label="📈 Canny High Threshold"
    ),
    gr.Number(value=-1, label="🌱 Seed (-1 for random)"),
    gr.Slider(
        minimum=256, maximum=1024, step=64, value=512, label="📐 Image Resolution"
    ),
    gr.Dropdown(
        choices=["lab", "yuv", "luminance"],
        value="luminance",
        label="🎨 Color Transfer Mode",
    ),
    gr.Slider(
        minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="🖌️ Color Transfer Strength"
    ),
]

outputs = [
    gr.Image(label="🎨 Color Transferred Image", type="pil"),
    gr.Image(label="🔍 Control Image (Canny Edge)", type="pil"),
    gr.Image(label="✨ Generated Image", type="pil")
]

examples = [
    [
        "example_images/mri_brain.jpg",
        "mri brain scan, good quality",
        "animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        20,
        7.5,
        1.0,
        100,
        200,
        -1,
        512,
        "luminance",
        1.0,
    ]
]

interface = gr.Interface(
    fn=process_image_with_controlnet,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=examples,
    cache_examples=False,
    allow_flagging="never",
)

if __name__ == "__main__":
    print("🚀 Launching Gradio UI...")
    interface.launch(server_name="0.0.0.0", server_port=7860)
