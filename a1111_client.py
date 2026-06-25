from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict

import requests
from PIL import Image


DEFAULT_A1111_BASE_URL = os.getenv("A1111_BASE_URL", "http://127.0.0.1:7861")
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted anatomy, extra limbs, duplicate body parts, bad hands, "
    "bad clothing details, text, watermark"
)


def _pil_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _round_to_multiple_of_64(value: int) -> int:
    return max(512, int(round(value / 64.0) * 64))


def build_img2img_payload(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    denoising_strength: float = 0.95,
    steps: int = 30,
    cfg_scale: float = 8.5,
    sampler_name: str = "DPM++ 2M Karras",
    checkpoint_name: str = "",
) -> Dict[str, Any]:
    width = _round_to_multiple_of_64(image.size[0])
    height = _round_to_multiple_of_64(image.size[1])

    payload: Dict[str, Any] = {
        "init_images": [_pil_to_base64(image.convert("RGB"))],
        "mask": _pil_to_base64(mask.convert("L")),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "sampler_name": sampler_name,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "resize_mode": 0,
        "mask_blur": 4,
        "inpainting_fill": 1,
        "inpaint_full_res": 1,
        "inpaint_full_res_padding": 32,
        "inpainting_mask_invert": 0,
        "include_init_images": False,
    }
    if checkpoint_name:
        payload["override_settings"] = {"sd_model_checkpoint": checkpoint_name}
    return payload


def run_a1111_img2img(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    base_url: str = DEFAULT_A1111_BASE_URL,
    checkpoint_name: str = "",
) -> Image.Image:
    payload = build_img2img_payload(
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        checkpoint_name=checkpoint_name,
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/sdapi/v1/img2img",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    encoded = data["images"][0]
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
