# Colab Clothing Recolor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-ready Gradio app that automatically finds skirt and socks regions in a photo, allows light mask correction, and recolors those garments without generative inpainting.

**Architecture:** A small Python app will separate responsibilities into segmentation, recoloring, utilities, and UI modules. The UI will orchestrate automatic mask prediction, optional correction, and deterministic recoloring while remaining easy to launch from a single Colab notebook.

**Tech Stack:** Python, Gradio, Pillow, NumPy, OpenCV, Transformers, Torch, Pytest

---

### Task 1: Scaffold the project

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `app.py`
- Create: `segmentation.py`
- Create: `recolor.py`
- Create: `utils.py`
- Create: `tests/test_recolor.py`
- Create: `tests/test_utils.py`
- Create: `colab.ipynb`

- [ ] Step 1: Write failing tests for utility and recolor behavior.
- [ ] Step 2: Run tests to verify they fail.
- [ ] Step 3: Implement minimal utility and recolor functions.
- [ ] Step 4: Run tests to verify they pass.
- [ ] Step 5: Add the remaining application files and notebook launcher.

### Task 2: Implement segmentation and UI integration

**Files:**
- Modify: `segmentation.py`
- Modify: `app.py`

- [ ] Step 1: Add a failing test for label-to-mask extraction helpers.
- [ ] Step 2: Run the targeted test to verify it fails if needed.
- [ ] Step 3: Implement segmentation helpers and configurable label mapping.
- [ ] Step 4: Integrate auto-mask preview and correction flow into Gradio.
- [ ] Step 5: Run the targeted tests again.

### Task 3: Verify the end-to-end flow

**Files:**
- Modify: `README.md`
- Modify: `colab.ipynb`

- [ ] Step 1: Run `pytest -q`.
- [ ] Step 2: Run a Python import smoke test for all modules.
- [ ] Step 3: Document exact Colab startup steps in `README.md`.
- [ ] Step 4: Update the notebook with clone/install/run cells.
