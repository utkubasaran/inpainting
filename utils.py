from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def combine_binary_masks(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required.")
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask.astype(np.uint8))
    return combined


def mask_to_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: int = 120,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[:, :, :3] = np.array(color, dtype=np.uint8)
    overlay[:, :, 3] = np.where(mask > 0, alpha, 0).astype(np.uint8)
    colored_mask = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(base, colored_mask)


def image_to_editor_value(mask: np.ndarray) -> Dict[str, Any]:
    mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
    return {"background": mask_image, "layers": [], "composite": mask_image}


def pil_mask_from_editor_value(editor_value: Dict[str, Any] | None, size: Tuple[int, int]) -> Image.Image:
    if not editor_value or editor_value.get("composite") is None:
        return Image.new("L", size, 0)
    composite = editor_value["composite"]
    if isinstance(composite, Image.Image):
        return composite.convert("L").resize(size)
    return Image.fromarray(np.asarray(composite).astype(np.uint8)).convert("L").resize(size)
