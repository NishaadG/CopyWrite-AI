"""Gradio web UI.

Local (laptop, no GPU):   python app.py --backend font
Colab / Kaggle (GPU):     python app.py --share          -> open the public *.gradio.live link
"""

import argparse
import json
import os
import tempfile

import gradio as gr

from hwgen.backends import get_generator
from hwgen.config import INK_COLORS, PAGE_SIZES_MM, PageConfig, hex_to_rgb
from hwgen.io_utils import read_text_file, save_pages
from hwgen.pipeline import generate_document
from hwgen.preprocess import prepare_style_refs

_GENERATORS = {}


def generator_for(name: str):
    if name not in _GENERATORS:  # load Emuru once and keep it in memory
        _GENERATORS[name] = get_generator(name)
    return _GENERATORS[name]


def _paths(files):
    if not files:
        return []
    return [f if isinstance(f, str) else getattr(f, "name", f) for f in files]


def preview_style(style_files, transcription, remove_ruling):
    try:
        refs = prepare_style_refs(_paths(style_files), transcription or "", remove_lines=remove_ruling)
        return [(r.image, r.text) for r in refs], f"OK - {len(refs)} style line(s) ready."
    except Exception as e:  # show the reason in the UI instead of crashing
        return [], f"Error: {e}"


def run(text, text_file, style_files, transcription, remove_ruling, backend,
        background, bg_photo, page_size, line_spacing, text_scale, jitter, margin_left,
        ink_name, ink_custom, watermark, seed, progress=gr.Progress()):
    if text_file:
        text = read_text_file(_paths([text_file])[0])
    if not text or not text.strip():
        raise gr.Error("Type some text or upload a .txt/.docx file.")
    try:
        refs = prepare_style_refs(_paths(style_files), transcription or "", remove_lines=remove_ruling)
    except ValueError as e:
        raise gr.Error(str(e))
    if background == "photo" and bg_photo is None:
        raise gr.Error("Upload a page photo, or pick ruled / plain / grid.")

    ink = hex_to_rgb(ink_custom) if ink_name == "custom" else INK_COLORS[ink_name]
    cfg = PageConfig(page_size=page_size, background=background, line_spacing_mm=line_spacing,
                     text_scale=text_scale, jitter=jitter, margin_left_mm=margin_left,
                     ink_color=ink, watermark=watermark, seed=int(seed) if seed is not None else None)

    progress(0.0, desc="Loading model (first run downloads ~3 GB)" if backend == "emuru" else "Starting")
    gen = generator_for(backend)
    result = generate_document(text, refs, gen, cfg, bg_photo,
                               progress=lambda p, m: progress(p, desc=m))
    out_dir = tempfile.mkdtemp(prefix="hwgen_")
    files = save_pages(result["pages"], out_dir, cfg.dpi)
    return result["pages"], files["pdf"], json.dumps(result["stats"], indent=2)


def build_ui(default_backend: str):
    with gr.Blocks(title="Handwritten Notes Generator") as demo:
        gr.Markdown("# ✍️ Handwritten Notes Generator\nType or upload notes, give a photo of a few "
                    "lines of your handwriting, and get pages written in your style.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("1. Content"):
                    text = gr.Textbox(label="Text to write", lines=10,
                                      placeholder="Paste your notes here. Each new line starts a new paragraph.")
                    text_file = gr.File(label="...or upload .txt / .docx", file_types=[".txt", ".md", ".docx"])
                with gr.Tab("2. Your handwriting"):
                    style_files = gr.File(label="Photo(s) of 1-5 lines of your handwriting",
                                          file_count="multiple", file_types=["image"])
                    transcription = gr.Textbox(label="What those lines say (one line per handwritten line, in order)",
                                               lines=4)
                    remove_ruling = gr.Checkbox(value=True, label="Remove notebook lines from the photo")
                    preview_btn = gr.Button("Check style lines")
                    style_gallery = gr.Gallery(label="Detected style lines", columns=1, height=240)
                    style_msg = gr.Markdown()
                with gr.Tab("3. Page"):
                    background = gr.Radio(["ruled", "plain", "grid", "photo"], value="ruled", label="Background")
                    bg_photo = gr.Image(label="Page photo (for 'photo' background)", type="pil")
                    page_size = gr.Dropdown(list(PAGE_SIZES_MM), value="A4", label="Page size")
                    line_spacing = gr.Slider(5, 14, value=8, step=0.5, label="Line spacing (mm)")
                    text_scale = gr.Slider(0.7, 1.6, value=1.15, step=0.05, label="Text size")
                    margin_left = gr.Slider(5, 40, value=25, step=1, label="Left margin (mm)")
                    jitter = gr.Slider(0, 2, value=1, step=0.1, label="Natural messiness")
                    with gr.Row():
                        ink_name = gr.Dropdown(list(INK_COLORS) + ["custom"], value="blue", label="Ink")
                        ink_custom = gr.ColorPicker(value="#1a2d8c", label="Custom ink")
                with gr.Tab("4. Engine"):
                    backend = gr.Radio(["emuru", "font"], value=default_backend,
                                       label="Generator (emuru = AI model, needs GPU for speed; font = no-AI baseline)")
                    watermark = gr.Checkbox(value=False, label="Add invisible watermark (optional)")
                    seed = gr.Number(value=42, precision=0, label="Random seed")
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                pages = gr.Gallery(label="Pages", columns=1, height=800)
                pdf = gr.File(label="Download PDF")
                stats = gr.Code(label="Run stats", language="json")

        preview_btn.click(preview_style, [style_files, transcription, remove_ruling], [style_gallery, style_msg])
        go.click(run, [text, text_file, style_files, transcription, remove_ruling, backend, background, bg_photo,
                       page_size, line_spacing, text_scale, jitter, margin_left, ink_name, ink_custom,
                       watermark, seed], [pages, pdf, stats])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public link (use this on Colab/Kaggle)")
    ap.add_argument("--backend", default=os.environ.get("HWGEN_BACKEND", "emuru"), choices=["emuru", "font"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    build_ui(args.backend).queue().launch(share=args.share, server_name=args.host, server_port=args.port)
