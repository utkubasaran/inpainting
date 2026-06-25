from __future__ import annotations

from typing import Dict


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
        f"replace the existing clothing with {cleaned_description}, "
        f"photo of a person wearing a realistic {part_description}, "
        f"natural folds, clearly visible garment change, matching pose, matching lighting"
    )
