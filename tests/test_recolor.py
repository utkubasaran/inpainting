from PIL import Image

from recolor import hex_to_rgb, recolor_region


def test_hex_to_rgb_supports_hash_prefixed_values():
    assert hex_to_rgb("#ff8800") == (255, 136, 0)


def test_hex_to_rgb_falls_back_for_empty_values():
    assert hex_to_rgb("") == (64, 64, 64)


def test_recolor_region_changes_only_masked_pixels():
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (120, 120, 120))
    image.putpixel((1, 0), (120, 120, 120))

    mask = Image.new("L", (2, 1), 0)
    mask.putpixel((0, 0), 255)

    recolored = recolor_region(image, mask, (255, 0, 0), strength=1.0)

    assert recolored.getpixel((0, 0)) != (120, 120, 120)
    assert recolored.getpixel((1, 0)) == (120, 120, 120)
