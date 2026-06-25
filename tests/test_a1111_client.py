from PIL import Image

from a1111_client import build_img2img_payload


def test_build_img2img_payload_includes_mask_and_prompt():
    image = Image.new("RGB", (128, 128), "black")
    mask = Image.new("L", (128, 128), 255)

    payload = build_img2img_payload(
        image=image,
        mask=mask,
        prompt="white blouse",
        negative_prompt="bad anatomy",
    )

    assert payload["prompt"] == "white blouse"
    assert payload["negative_prompt"] == "bad anatomy"
    assert payload["inpainting_fill"] == 1
    assert payload["mask_blur"] == 4
    assert payload["init_images"]
    assert payload["mask"]


def test_build_img2img_payload_uses_dimensions_rounded_to_64():
    image = Image.new("RGB", (510, 770), "black")
    mask = Image.new("L", (510, 770), 255)

    payload = build_img2img_payload(
        image=image,
        mask=mask,
        prompt="navy cardigan",
        negative_prompt="bad anatomy",
    )

    assert payload["width"] % 64 == 0
    assert payload["height"] % 64 == 0
