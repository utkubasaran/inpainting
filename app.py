from __future__ import annotations

from typing import Any, Dict, Tuple

from PIL import Image

from inpaint import DEFAULT_INPAINT_MODEL, build_inpaint_prompt, run_inpaint
from segmentation import DEFAULT_MODEL_NAME, detect_garment_masks
from utils import image_to_editor_value, mask_to_overlay, pil_mask_from_editor_value

try:
    import gradio as gr
except ImportError:  # pragma: no cover
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
    "bag": (0, 200, 140),
    "scarf": (180, 0, 255),
    "belt": (120, 120, 120),
    "glasses": (30, 30, 30),
}


def _build_part_choices(masks: dict[str, Any]) -> list[str]:
    return [part for part, mask in masks.items() if mask.max() > 0]


def auto_detect_masks(
    image: Image.Image,
    segmentation_model_name: str,
) -> Tuple[Image.Image, EditorValue, Any, str, dict[str, Any]]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Fotoğraf yükledikten sonra algılama çalışır.")

    masks = detect_garment_masks(image, model_name=segmentation_model_name or DEFAULT_MODEL_NAME)
    choices = _build_part_choices(masks)
    if not choices:
        raise gr.Error("Algılanan düzenlenebilir bir kıyafet parçası bulunamadı.")

    selected_part = choices[0]
    overlay = mask_to_overlay(image, masks[selected_part], MASK_COLORS.get(selected_part, (255, 0, 0)))
    status = "Maske hazır. İstersen küçük düzeltme yap, sonra kıyafeti yazıp üret."

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
    if image is None or not masks_state or selected_part not in masks_state:
        raise gr.Error("Önce fotoğraf yükleyip maske algılat.")

    mask = masks_state[selected_part]
    overlay = mask_to_overlay(image, mask, MASK_COLORS.get(selected_part, (255, 0, 0)))
    return overlay, image_to_editor_value(mask)


def generate_edit(
    image: Image.Image,
    selected_part: str,
    part_editor: EditorValue,
    garment_description: str,
    inpaint_model_name: str,
) -> Tuple[Image.Image, str]:
    if gr is None:
        raise ImportError("Gradio is required to use the web UI.")
    if image is None:
        raise gr.Error("Fotoğraf yüklemeden üretim yapılamaz.")

    working = image.convert("RGB")
    selected_mask = pil_mask_from_editor_value(part_editor, working.size)
    prompt = build_inpaint_prompt(selected_part, garment_description)
    result = run_inpaint(
        image=working,
        mask=selected_mask,
        prompt=prompt,
        model_name=inpaint_model_name or DEFAULT_INPAINT_MODEL,
    )
    return result, prompt


def build_demo() -> gr.Blocks:
    if gr is None:
        raise ImportError("Gradio is required. Install requirements.txt before launching the app.")

    with gr.Blocks(title="Kiyafet Inpainting") as demo:
        masks_state = gr.State({})

        gr.Markdown(
            """
            # Kiyafet Inpainting
            Fotoğrafı yükle, düzenlemek istediğin parçayı seç, örneğin `white blouse` veya `navy cardigan`
            yaz ve sonucu üret.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Fotoğraf", height=520)
                detect_button = gr.Button("Maskeyi otomatik algıla", variant="primary")
                selected_part = gr.Dropdown(choices=[], label="Parça")
                garment_description = gr.Textbox(
                    label="Ne olsun?",
                    placeholder="white blouse, navy cardigan, white shirt",
                )
                generate_button = gr.Button("Sonucu üret", variant="primary")
                status_text = gr.Textbox(label="Durum", interactive=False)
                with gr.Accordion("Gelişmiş", open=False):
                    segmentation_model_name = gr.Textbox(
                        value=DEFAULT_MODEL_NAME,
                        label="Segmentasyon Modeli",
                    )
                    inpaint_model_name = gr.Textbox(
                        value=DEFAULT_INPAINT_MODEL,
                        label="Inpaint Modeli",
                    )
                    prompt_preview = gr.Textbox(label="Kullanılan Prompt", interactive=False)

            with gr.Column(scale=2):
                mask_preview = gr.Image(type="pil", label="Maske Önizleme", height=520)
                mask_editor = gr.ImageEditor(label="Gerekirse maskeyi düzelt", height=520)

            with gr.Column(scale=2):
                result_output = gr.Image(type="pil", label="Sonuç", height=520)

        detect_button.click(
            fn=auto_detect_masks,
            inputs=[image_input, segmentation_model_name],
            outputs=[mask_preview, mask_editor, selected_part, status_text, masks_state],
        )
        selected_part.change(
            fn=update_selected_part,
            inputs=[image_input, selected_part, masks_state],
            outputs=[mask_preview, mask_editor],
        )
        generate_button.click(
            fn=generate_edit,
            inputs=[image_input, selected_part, mask_editor, garment_description, inpaint_model_name],
            outputs=[result_output, prompt_preview],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(debug=True, share=True)
