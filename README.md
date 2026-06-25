# Clothing Inpainting for Colab

This project builds a small `Gradio` app for changing a selected clothing region in a photo with text prompts such as `white blouse`, `white shirt`, or `navy cardigan`. It uses automatic human parsing to find the region, then runs text-guided inpainting on only that masked area.

## Features

- Google Colab friendly startup flow
- Automatic segmentation for top, skirt, dress, pants, shoes, and similar parts
- Optional mask touch-up in the browser
- Text-guided inpainting for changing garment type and color
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
- Default inpainting checkpoint is `stable-diffusion-v1-5/stable-diffusion-inpainting`.
- If a different checkpoint uses different label IDs, override with environment variables:

```bash
set SKIRT_LABELS=5
set SOCK_LABELS=7,8
```

- The inpainting pipeline can change clothing style and color, but its quality still depends on the segmentation mask and the prompt.
