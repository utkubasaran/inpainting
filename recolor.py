from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


RgbColor = Tuple[int, int, int]


def hex_to_rgb(value: str) -> RgbColor:
    if not value:
        return (64, 64, 64)
    normalized = value.strip().lstrip("#")
    if len(normalized) != 6:
        return (64, 64, 64)
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def recolor_region(
    image: Image.Image,
    mask: Image.Image,
    target_rgb: RgbColor,
    strength: float = 0.85,
) -> Image.Image:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    mask_array = np.expand_dims(mask_array * float(np.clip(strength, 0.0, 1.0)), axis=-1)

    target_array = np.zeros_like(image_array)
    target_array[:, :] = np.array(target_rgb, dtype=np.float32)

    recolored = image_array * (1.0 - mask_array) + target_array * mask_array
    recolored = np.clip(recolored, 0, 255).astype(np.uint8)
    return Image.fromarray(recolored, mode="RGB")
