# Clothing Recolor for Colab

This project builds a small `Gradio` app for recoloring skirt and socks areas in a photo while keeping the original pose and texture. It avoids manual inpainting by using automatic human parsing, then lets you make only small mask corrections when the auto-detection is imperfect.

## Features

- Google Colab friendly startup flow
- Automatic segmentation for skirt and socks regions
- Optional mask touch-up in the browser
- Deterministic recoloring instead of generative repainting
- Before/after preview and download support through Gradio

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Colab Setup

1. Create a new Colab notebook or upload [colab.ipynb](/C:/Users/Utku/Desktop/im2im/colab.ipynb).
2. Run the clone cell and replace `YOUR_GITHUB_REPO_URL` with your repo URL.
3. Run the install cell.
4. Run the launch cell.
5. Open the Gradio public URL shown in the output.

## Notes

- Default segmentation checkpoint is `mattmdjaga/segformer_b2_clothes`.
- If a different checkpoint uses different label IDs, override with environment variables:

```bash
set SKIRT_LABELS=5
set SOCK_LABELS=7,8
```

- The recolor pipeline preserves texture better than inpainting, but it does not invent new fabric structure.
