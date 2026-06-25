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

PART_ALIASES: dict[str, tuple[str, ...]] = {
    "hat": ("hat",),
    "hair": ("hair",),
    "glasses": ("sunglasses", "glasses"),
    "top": ("upper clothes", "upper-clothes", "top", "shirt", "tshirt", "tee", "blouse"),
    "skirt": ("skirt",),
    "pants": ("pants", "trousers", "jeans"),
    "dress": ("dress", "jumpsuit", "jumpsuits"),
    "belt": ("belt",),
    "shoes": ("shoe", "shoes", "boot", "boots", "sneaker", "sneakers"),
    "face": ("face",),
    "bag": ("bag", "handbag", "purse"),
    "scarf": ("scarf",),
    "socks": ("sock", "socks", "stocking", "stockings", "hosiery", "leg warmer"),
}

PART_ENV_DEFAULTS: dict[str, tuple[int, ...]] = {
    "hat": (1,),
    "hair": (2,),
    "glasses": tuple(),
    "top": (3, 4),
    "skirt": (5,),
    "pants": (6,),
    "dress": (7,),
    "belt": (8,),
    "shoes": (9, 10),
    "face": (11,),
    "bag": (16,),
    "scarf": (17,),
    "socks": tuple(),
}


@dataclass(frozen=True)
class GarmentLabelMap:
    top_labels: tuple[int, ...] = tuple()
    skirt_labels: tuple[int, ...] = tuple()
    sock_labels: tuple[int, ...] = tuple()
    dress_labels: tuple[int, ...] = tuple()
    pants_labels: tuple[int, ...] = tuple()
    hat_labels: tuple[int, ...] = tuple()
    hair_labels: tuple[int, ...] = tuple()
    face_labels: tuple[int, ...] = tuple()
    shoe_labels: tuple[int, ...] = tuple()
    bag_labels: tuple[int, ...] = tuple()
    scarf_labels: tuple[int, ...] = tuple()
    belt_labels: tuple[int, ...] = tuple()
    glasses_labels: tuple[int, ...] = tuple()

    @property
    def part_labels(self) -> dict[str, tuple[int, ...]]:
        return {
            "top": self.top_labels,
            "skirt": self.skirt_labels,
            "socks": self.sock_labels,
            "dress": self.dress_labels,
            "pants": self.pants_labels,
            "hat": self.hat_labels,
            "hair": self.hair_labels,
            "face": self.face_labels,
            "shoes": self.shoe_labels,
            "bag": self.bag_labels,
            "scarf": self.scarf_labels,
            "belt": self.belt_labels,
            "glasses": self.glasses_labels,
        }


def _parse_env_labels(name: str, default: Iterable[int]) -> tuple[int, ...]:
    raw_value = os.getenv(name)
    if not raw_value:
        return tuple(default)
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def extract_garment_masks(
    parsing_map: np.ndarray,
    label_map: GarmentLabelMap,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for part_name, label_ids in label_map.part_labels.items():
        if not label_ids:
            masks[part_name] = np.zeros_like(parsing_map, dtype=np.uint8)
            continue
        masks[part_name] = np.isin(parsing_map, label_ids).astype(np.uint8) * 255
    return masks


def infer_label_map(id2label: Dict[int, str]) -> GarmentLabelMap:
    part_labels: dict[str, list[int]] = {part_name: [] for part_name in PART_ALIASES}

    for label_id, label_name in id2label.items():
        normalized = label_name.lower().replace("-", " ").replace("_", " ")
        for part_name, aliases in PART_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                part_labels[part_name].append(int(label_id))

    resolved: dict[str, tuple[int, ...]] = {}
    for part_name, default_ids in PART_ENV_DEFAULTS.items():
        env_name = f"{part_name.upper()}_LABELS"
        labels = tuple(part_labels[part_name])
        if not labels:
            labels = _parse_env_labels(env_name, default_ids)
        resolved[part_name] = labels

    return GarmentLabelMap(
        top_labels=resolved["top"],
        skirt_labels=resolved["skirt"],
        sock_labels=resolved["socks"],
        dress_labels=resolved["dress"],
        pants_labels=resolved["pants"],
        hat_labels=resolved["hat"],
        hair_labels=resolved["hair"],
        face_labels=resolved["face"],
        shoe_labels=resolved["shoes"],
        bag_labels=resolved["bag"],
        scarf_labels=resolved["scarf"],
        belt_labels=resolved["belt"],
        glasses_labels=resolved["glasses"],
    )


def get_available_parts(label_map: GarmentLabelMap) -> list[str]:
    ordered_parts = [
        "top",
        "skirt",
        "dress",
        "pants",
        "shoes",
        "socks",
        "hat",
        "hair",
        "face",
        "bag",
        "scarf",
        "belt",
        "glasses",
    ]
    return [part_name for part_name in ordered_parts if label_map.part_labels.get(part_name)]


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
