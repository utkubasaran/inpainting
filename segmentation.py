from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
except ImportError:  # pragma: no cover - exercised in runtime environments without ML deps
    AutoImageProcessor = None
    AutoModelForSemanticSegmentation = None
    torch = None


DEFAULT_MODEL_NAME = os.getenv(
    "SEGMENTATION_MODEL_NAME",
    "mattmdjaga/segformer_b2_clothes",
)


@dataclass(frozen=True)
class GarmentLabelMap:
    skirt_labels: tuple[int, ...]
    sock_labels: tuple[int, ...]


def _parse_env_labels(name: str, default: Iterable[int]) -> tuple[int, ...]:
    raw_value = os.getenv(name)
    if not raw_value:
        return tuple(default)
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def extract_garment_masks(
    parsing_map: np.ndarray,
    label_map: GarmentLabelMap,
) -> Dict[str, np.ndarray]:
    skirt_mask = np.isin(parsing_map, label_map.skirt_labels).astype(np.uint8) * 255
    sock_mask = np.isin(parsing_map, label_map.sock_labels).astype(np.uint8) * 255
    return {"skirt": skirt_mask, "socks": sock_mask}


def infer_label_map(id2label: Dict[int, str]) -> GarmentLabelMap:
    skirt_labels = []
    sock_labels = []

    for label_id, label_name in id2label.items():
        normalized = label_name.lower().replace("-", " ").replace("_", " ")
        if any(term in normalized for term in ("skirt", "dress")):
            skirt_labels.append(int(label_id))
        if any(term in normalized for term in ("sock", "stocking", "leg warmer")):
            sock_labels.append(int(label_id))

    if not skirt_labels:
        skirt_labels = list(_parse_env_labels("SKIRT_LABELS", (5,)))
    if not sock_labels:
        sock_labels = list(_parse_env_labels("SOCK_LABELS", (7, 8)))

    return GarmentLabelMap(tuple(skirt_labels), tuple(sock_labels))


@lru_cache(maxsize=1)
def load_segmentation_components(model_name: str = DEFAULT_MODEL_NAME):
    if AutoImageProcessor is None or AutoModelForSemanticSegmentation is None or torch is None:
        raise ImportError(
            "Transformers and torch are required. Install requirements.txt before running the app."
        )

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    return processor, model


def predict_parsing_map(image: Image.Image, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    processor, model = load_segmentation_components(model_name)
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def detect_garment_masks(
    image: Image.Image,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, np.ndarray]:
    parsing_map = predict_parsing_map(image, model_name)
    _, model = load_segmentation_components(model_name)
    label_map = infer_label_map(model.config.id2label)
    return extract_garment_masks(parsing_map, label_map)
