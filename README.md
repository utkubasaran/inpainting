# Clothing Inpainting for Colab + AUTOMATIC1111

This project builds a small `Gradio` app for changing a selected clothing region in a photo with text prompts such as `white blouse`, `white shirt`, or `navy cardigan`. It uses automatic human parsing to find the region, then sends the masked edit request to a locally running `AUTOMATIC1111` API inside Colab.

## Features

- Google Colab friendly startup flow
- Automatic segmentation for top, skirt, dress, pants, shoes, and similar parts
- Optional mask touch-up in the browser
- Text-guided inpainting for changing garment type and color through `AUTOMATIC1111`
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
4. Run the `AUTOMATIC1111` startup cell and wait for the API to come up.
5. Run the app launch cell.
6. Open the Gradio public URL shown in the output.

## Notes

- Default segmentation checkpoint is `mattmdjaga/segformer_b2_clothes`.
- The app expects `AUTOMATIC1111` to be reachable at `http://127.0.0.1:7861` by default.
- You can optionally pass a checkpoint name in the advanced section if multiple models are installed in A1111.
- If a different checkpoint uses different label IDs, override with environment variables:

```bash
set SKIRT_LABELS=5
set SOCK_LABELS=7,8
```

- `white blouse`, `white shirt`, and `navy cardigan` are realistic prompts for this setup.
- `white bra` is much harder because it requires the model to remove the outer garment rather than replace it with another outer garment.
