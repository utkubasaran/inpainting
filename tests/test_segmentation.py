import numpy as np

from segmentation import GarmentLabelMap, extract_garment_masks


def test_extract_garment_masks_returns_expected_regions():
    parsing_map = np.array(
        [
            [0, 5, 0],
            [7, 0, 8],
        ],
        dtype=np.uint8,
    )
    label_map = GarmentLabelMap(skirt_labels=(5,), sock_labels=(7, 8))

    masks = extract_garment_masks(parsing_map, label_map)

    assert masks["skirt"].tolist() == [[0, 255, 0], [0, 0, 0]]
    assert masks["socks"].tolist() == [[0, 0, 0], [255, 0, 255]]
