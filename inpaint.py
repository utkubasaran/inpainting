from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

from PIL import Image, ImageFilter

try:
    import torch
    from diffusers import AutoPipelineForInpainting
except ImportError:  # pragma: no cover
    AutoPipelineForInpainting = None
    torch = None


DEFAULT_INPAINT_MODEL = os.getenv(
    "INPAINT_MODEL_NAME",
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
)

PART_DESCRIPTIONS: Dict[str, str] = {
    "top": "upper body garment",
    "skirt": "skirt",
    "dress": "dress",
    "pants": "pants",
    "shoes": "shoes",
    "socks": "socks or stockings",
    "hat": "hat",
    "hair": "hair style or hair color",
    "bag": "bag",
    "scarf": "scarf",
    "belt": "belt",
    "glasses": "glasses",
}


def build_inpaint_prompt(selected_part: str, garment_description: str) -> str:
    cleaned_description = " ".join((garment_description or "").split()).strip()
    if not cleaned_description:
        cleaned_description = "clean well-fitted clothing"

    part_description = PART_DESCRIPTIONS.get(selected_part, selected_part)
    return (
        f"photo of a person wearing {cleaned_description}, "
        f"realistic {part_description}, natural folds, matching pose, matching lighting"
    )


def soften_mask(mask: Image.Image, blur_radius: int = 6) -> Image.Image:
    expanded = mask.convert("L").filter(ImageFilter.MaxFilter(size=5))
    return expanded.filter(ImageFilter.GaussianBlur(radius=blur_radius))


@lru_cache(maxsize=1)
def load_inpaint_pipeline(model_name: str = DEFAULT_INPAINT_MODEL):
    if AutoPipelineForInpainting is None or torch is None:
        raise ImportError(
            "diffusers and torch are required. Install requirements.txt before running the app."
        )

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "extra limbs, distorted anatomy, blurry, low quality, mismatched clothing",
    strength: float = 0.95,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    model_name: str = DEFAULT_INPAINT_MODEL,
) -> Image.Image:
    pipe = load_inpaint_pipeline(model_name)
    resized_image = image.convert("RGB").resize((512, 512))
    resized_mask = soften_mask(mask).resize((512, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=resized_image,
        mask_image=resized_mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return result.resize(image.size)
