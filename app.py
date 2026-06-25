from __future__ import annotations

from typing import Any, Dict, Tuple

from PIL import Image

from recolor import hex_to_rgb, recolor_region
from segmentation import DEFAULT_MODEL_NAME, detect_garment_masks
from utils import image_to_editor_value, mask_to_overlay, pil_mask_from_editor_value

try:
    import gradio as gr
except ImportError:  # pragma: no cover - dependency availability varies outside Colab
    gr = None


EditorValue = Dict[str, Any]
MASK_COLORS: dict[str, tuple[int, int, int]] = {
    "top": (255, 140, 0),
    "skirt": (255, 0, 64),
    "dress": (255, 64, 180),
    "pants": (100, 180, 255),
    "shoes": (0, 160, 255),
    "socks": (140, 220, 255),
    "hat": (240, 220, 0),
    "hair": (170, 110, 60),
    "face": (255, 200, 180),
    "bag": (0, 200, 140),
    "scarf": (180, 0, 255),
    "belt": (120, 120, 120),
    "glasses": (30, 30, 30),
}
PART_LABELS: dict[str, str] = {
    "top": "Top / T-shirt",
    "skirt": "Skirt",
    "dress": "Dress",
    "pants": "Pants",
    "shoes": "Shoes",
    "socks": "Socks / Stockings",
    "hat": "Hat",
    "hair": "Hair",
    "face": "Face",
    "bag": "Bag",
    "scarf": "Scarf",
    "belt": "Belt",
    "glasses": "Glasses",
}


def _empty_editor_value() -> EditorValue:
    return {"background": None, "layers": [], "composite": None}


def _build_part_choices(masks: dict[str, Any]) -> list[str]:
    choices = [part for part, mask in masks.items() if mask.max() > 0]
    return choices or ["top"]


def auto_detect_masks(
    image: Image.Image,
    model_name: str,
) -> Tuple[Image.Image, EditorValue, Any, str, dict[str, Any]]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Once a photo is uploaded, mask detection can run.")

    masks = detect_garment_masks(image, model_name=model_name or DEFAULT_MODEL_NAME)
    choices = _build_part_choices(masks)
    selected_part = choices[0]
    overlay = mask_to_overlay(image, masks[selected_part], MASK_COLORS.get(selected_part, (255, 0, 0)))

    missing = [PART_LABELS.get(part, part) for part, mask in masks.items() if mask.max() == 0]
    status = "Detected selectable parts. Pick one from the dropdown."
    if missing and len(missing) != len(masks):
        status += " Missing: " + ", ".join(missing[:6])

    return (
        overlay,
        image_to_editor_value(masks[selected_part]),
        gr.update(choices=choices, value=selected_part),
        status,
        masks,
    )


def update_selected_part(
    image: Image.Image,
    selected_part: str,
    masks_state: dict[str, Any],
) -> Tuple[Image.Image, EditorValue]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Upload a photo first.")
    if not masks_state or selected_part not in masks_state:
        raise gr.Error("Run auto detection first.")

    mask = masks_state[selected_part]
    overlay = mask_to_overlay(image, mask, MASK_COLORS.get(selected_part, (255, 0, 0)))
    return overlay, image_to_editor_value(mask)


def apply_recolor(
    image: Image.Image,
    selected_part: str,
    part_editor: EditorValue,
    target_color: str,
    strength: float,
) -> Tuple[Image.Image, Image.Image]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Upload a photo before recoloring.")

    working = image.convert("RGB")
    selected_mask = pil_mask_from_editor_value(part_editor, working.size)
    result = recolor_region(working, selected_mask, hex_to_rgb(target_color), strength=strength)
    return working, result


def build_demo() -> gr.Blocks:
    if gr is None:
        raise ImportError("Gradio is required. Install requirements.txt before launching the app.")
    with gr.Blocks(title="Clothing Recolor for Colab") as demo:
        masks_state = gr.State({})

        gr.Markdown(
            """
            # Clothing Recolor
            Upload a photo, auto-detect human parts and clothes, choose the part you want,
            make a small mask correction if needed, then recolor only that selected region.
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Person Photo")
                model_name = gr.Textbox(
                    value=DEFAULT_MODEL_NAME,
                    label="Segmentation Model",
                    info="Default is an ATR-based clothes parsing checkpoint.",
                )
                detect_button = gr.Button("Auto Detect Parts", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)
                selected_part = gr.Dropdown(
                    choices=[],
                    label="Detected Part",
                    info="Examples: top, skirt, dress, shoes, hat, hair.",
                )
                target_color = gr.ColorPicker(label="Target Color", value="#404040")
                strength = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Recolor Strength")
                apply_button = gr.Button("Apply Recolor")

            with gr.Column():
                selected_overlay = gr.Image(type="pil", label="Selected Part Preview")
                part_editor = gr.ImageEditor(label="Adjust Selected Mask")

            with gr.Column():
                original_output = gr.Image(type="pil", label="Original")
                result_output = gr.Image(type="pil", label="Recolored Result")

        detect_button.click(
            fn=auto_detect_masks,
            inputs=[image_input, model_name],
            outputs=[selected_overlay, part_editor, selected_part, status_text, masks_state],
        )
        selected_part.change(
            fn=update_selected_part,
            inputs=[image_input, selected_part, masks_state],
            outputs=[selected_overlay, part_editor],
        )
        apply_button.click(
            fn=apply_recolor,
            inputs=[image_input, selected_part, part_editor, target_color, strength],
            outputs=[original_output, result_output],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(debug=True, share=True)
