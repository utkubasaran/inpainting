import numpy as np

from segmentation import (
    GarmentLabelMap,
    extract_garment_masks,
    get_available_parts,
    infer_label_map,
)


def test_extract_garment_masks_returns_expected_regions():
    parsing_map = np.array(
        [
            [0, 5, 3],
            [7, 0, 8],
        ],
        dtype=np.uint8,
    )
    label_map = GarmentLabelMap(top_labels=(3,), skirt_labels=(5,), sock_labels=(7, 8))

    masks = extract_garment_masks(parsing_map, label_map)

    assert masks["top"].tolist() == [[0, 0, 255], [0, 0, 0]]
    assert masks["skirt"].tolist() == [[0, 255, 0], [0, 0, 0]]
    assert masks["socks"].tolist() == [[0, 0, 0], [255, 0, 255]]


def test_infer_label_map_detects_upper_clothes_labels():
    label_map = infer_label_map(
        {
            0: "background",
            3: "upper-clothes",
            5: "skirt",
            7: "left_sock",
            8: "right_stocking",
        }
    )

    assert label_map.top_labels == (3,)
    assert label_map.skirt_labels == (5,)
    assert label_map.sock_labels == (7, 8)


def test_get_available_parts_lists_detected_regions():
    label_map = infer_label_map(
        {
            0: "background",
            1: "hat",
            2: "hair",
            3: "upper-clothes",
            5: "skirt",
            9: "left-shoe",
            10: "right-shoe",
            17: "scarf",
        }
    )

    parts = get_available_parts(label_map)

    assert "hat" in parts
    assert "hair" in parts
    assert "top" in parts
    assert "skirt" in parts
    assert "shoes" in parts
    assert "scarf" in parts
