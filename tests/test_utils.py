import numpy as np
from PIL import Image

from utils import combine_binary_masks, mask_to_overlay


def test_combine_binary_masks_uses_pixelwise_maximum():
    first = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    second = np.array([[0, 0], [255, 0]], dtype=np.uint8)

    combined = combine_binary_masks(first, second)

    assert combined.tolist() == [[0, 255], [255, 0]]


def test_mask_to_overlay_returns_rgba_preview():
    image = Image.new("RGB", (1, 1), (10, 20, 30))
    mask = np.array([[255]], dtype=np.uint8)

    overlay = mask_to_overlay(image, mask, (255, 0, 0))

    assert overlay.mode == "RGBA"
    assert overlay.size == (1, 1)
