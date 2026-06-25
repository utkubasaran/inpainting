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


def auto_detect_masks(
    image: Image.Image,
    model_name: str,
) -> Tuple[Image.Image, Image.Image, Image.Image, EditorValue, EditorValue, EditorValue, str]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Once a photo is uploaded, mask detection can run.")

    masks = detect_garment_masks(image, model_name=model_name or DEFAULT_MODEL_NAME)
    top_mask = masks["top"]
    skirt_mask = masks["skirt"]
    socks_mask = masks["socks"]

    top_overlay = mask_to_overlay(image, top_mask, (255, 140, 0))
    skirt_overlay = mask_to_overlay(image, skirt_mask, (255, 0, 64))
    socks_overlay = mask_to_overlay(image, socks_mask, (0, 160, 255))

    status = []
    if top_mask.max() == 0:
        status.append("Top mask not found")
    if skirt_mask.max() == 0:
        status.append("Skirt mask not found")
    if socks_mask.max() == 0:
        status.append("Socks mask not found")
    if not status:
        status.append("Masks detected. Touch up only if needed.")

    return (
        top_overlay,
        skirt_overlay,
        socks_overlay,
        image_to_editor_value(top_mask),
        image_to_editor_value(skirt_mask),
        image_to_editor_value(socks_mask),
        " | ".join(status),
    )


def apply_recolor(
    image: Image.Image,
    top_editor: EditorValue,
    skirt_editor: EditorValue,
    socks_editor: EditorValue,
    top_color: str,
    skirt_color: str,
    socks_color: str,
    strength: float,
) -> Tuple[Image.Image, Image.Image]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Upload a photo before recoloring.")

    working = image.convert("RGB")
    top_mask = pil_mask_from_editor_value(top_editor, working.size)
    skirt_mask = pil_mask_from_editor_value(skirt_editor, working.size)
    socks_mask = pil_mask_from_editor_value(socks_editor, working.size)

    result = recolor_region(working, top_mask, hex_to_rgb(top_color), strength=strength)
    result = recolor_region(result, skirt_mask, hex_to_rgb(skirt_color), strength=strength)
    result = recolor_region(result, socks_mask, hex_to_rgb(socks_color), strength=strength)
    return working, result


def build_demo() -> gr.Blocks:
    if gr is None:
        raise ImportError("Gradio is required. Install requirements.txt before launching the app.")
    with gr.Blocks(title="Clothing Recolor for Colab") as demo:
        gr.Markdown(
            """
            # Clothing Recolor
            Upload a photo, auto-detect top, skirt, and socks, make small mask corrections if needed,
            then recolor the garments without generative inpainting.
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Person Photo")
                model_name = gr.Textbox(
                    value=DEFAULT_MODEL_NAME,
                    label="Segmentation Model",
                    info="Can be replaced with any compatible segmentation checkpoint.",
                )
                detect_button = gr.Button("Auto Detect Masks", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)
                top_color = gr.ColorPicker(label="Top Color", value="#404040")
                skirt_color = gr.ColorPicker(label="Skirt Color", value="#202020")
                socks_color = gr.ColorPicker(label="Socks Color", value="#f5f5f5")
                strength = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Recolor Strength")
                apply_button = gr.Button("Apply Recolor")

            with gr.Column():
                top_overlay = gr.Image(type="pil", label="Auto Top Mask Preview")
                skirt_overlay = gr.Image(type="pil", label="Auto Skirt Mask Preview")
                socks_overlay = gr.Image(type="pil", label="Auto Socks Mask Preview")
                top_editor = gr.ImageEditor(label="Adjust Top Mask")
                skirt_editor = gr.ImageEditor(label="Adjust Skirt Mask")
                socks_editor = gr.ImageEditor(label="Adjust Socks Mask")

            with gr.Column():
                original_output = gr.Image(type="pil", label="Original")
                result_output = gr.Image(type="pil", label="Recolored Result")

        detect_button.click(
            fn=auto_detect_masks,
            inputs=[image_input, model_name],
            outputs=[top_overlay, skirt_overlay, socks_overlay, top_editor, skirt_editor, socks_editor, status_text],
        )
        apply_button.click(
            fn=apply_recolor,
            inputs=[image_input, top_editor, skirt_editor, socks_editor, top_color, skirt_color, socks_color, strength],
            outputs=[original_output, result_output],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(debug=True, share=True)
