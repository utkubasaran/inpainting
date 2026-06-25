# Colab Clothing Recolor Tool Design

**Goal:** Build a Google Colab friendly web app that uploads a person photo, auto-detects skirt and socks/stockings regions, lets the user lightly correct the masks, and recolors those garments while keeping pose and texture intact.

## Approach

The tool will run as a lightweight `Gradio` app inside Colab. Instead of generative inpainting, it will use human parsing to obtain garment masks and then apply deterministic recoloring in the selected regions. This keeps the pose unchanged, preserves fabric texture better, and avoids heavy manual masking.

## Components

- `app.py`: Gradio interface and app lifecycle.
- `segmentation.py`: automatic person parsing and garment mask extraction.
- `recolor.py`: hue-preserving recolor pipeline with target color blending.
- `utils.py`: image conversions, overlays, and mask helpers.
- `tests/`: unit tests for recolor math and mask helpers.
- `colab.ipynb`: notebook launcher for clone/install/run flow.

## Data Flow

1. User uploads an image.
2. Segmentation module predicts semantic garment labels.
3. App derives `skirt` and `socks` masks from those labels.
4. UI shows preview overlays for auto-selected regions.
5. User optionally refines masks with a light brush editor.
6. Recolor pipeline applies independent target colors to skirt and socks.
7. Result is shown side-by-side and can be downloaded.

## Model Choice

Use a parsing model available through Hugging Face `transformers` for easy Colab installation. Primary target is SegFormer clothing/human parsing style checkpoints; the code should keep label mapping configurable so the exact model can be swapped if a better checkpoint is preferred later.

## Error Handling

- If no person-like garment labels are found, show a clear UI error.
- If `skirt` or `socks` masks are empty, keep the original image and report which region was missing.
- If the model fails to load in Colab, surface install and runtime guidance instead of crashing silently.

## Testing

- Unit tests for hex-to-RGB conversion and recolor function behavior.
- Unit tests for mask extraction helpers.
- Smoke verification by importing modules and rendering the Gradio app locally in a non-interactive check.
