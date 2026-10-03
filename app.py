"""CopyWrite AI - Gradio web UI.

Local (laptop, no GPU):   python app.py                 -> http://127.0.0.1:7860 (font baseline)
Colab / Kaggle (GPU):     python app.py --share         -> open the public *.gradio.live link
The AI model (emuru) is picked automatically when a GPU is present; override with --backend.
"""

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
import traceback

import gradio as gr
from PIL import Image

from copywrite.backends import get_generator
from copywrite.config import INK_COLORS, PAGE_SIZES_MM, PageConfig, hex_to_rgb
from copywrite.io_utils import read_text_file, save_pages
from copywrite.pipeline import generate_document
from copywrite.preprocess import build_refs, detect_style_lines

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_TEXT = read_text_file(os.path.join(HERE, "samples", "text", "sample_notes.txt")).strip()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6  # Gradio 6 moved theme/css from Blocks() to launch()

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
CSS = """
.gradio-container {max-width: 1320px !important; margin: 0 auto !important;}
#hero {text-align: center; padding: 8px 0 4px;}
#hero h1 {font-size: 2.1rem; margin-bottom: 4px;}
#hero p {opacity: .8; margin: 2px 0;}
.step h3 {margin: 14px 0 2px;}
#write-btn {font-size: 1.15rem; min-height: 54px;}
#status {min-height: 28px;}
/* handwriting samples are wide single lines: show them as full-width rows, not squares */
#lib-gallery .grid-container {grid-template-columns: 1fr !important; grid-template-rows: none !important;
                              grid-auto-rows: auto !important;}
#lib-gallery .thumbnail-item {aspect-ratio: auto !important; height: auto !important; background: #fff;}
#lib-gallery .thumbnail-item img {width: 100% !important; height: auto !important; object-fit: contain !important;}
"""

_GENERATORS = {}
_LIBRARY = {"refs": None, "error": None}
_LIBRARY_LOCK = threading.Lock()


def generator_for(name: str):
    if name not in _GENERATORS:  # load the model once and keep it in memory
        _GENERATORS[name] = get_generator(name)
    return _GENERATORS[name]


def ocr_for_checking():
    """TrOCR, used to read back and verify every generated piece; None if it can't be loaded."""
    try:
        from copywrite.ocr import get_ocr

        return get_ocr()
    except Exception:
        return None


def gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _paths(files):
    if not files:
        return []
    return [f if isinstance(f, str) else getattr(f, "name", f) for f in files]


# ---------------------------------------------------------------- handwriting library ----
def library_refs():
    with _LIBRARY_LOCK:  # also preloaded in a background thread at startup
        if _LIBRARY["refs"] is None and _LIBRARY["error"] is None:
            try:
                from copywrite.library import load_library

                _LIBRARY["refs"] = load_library()
            except Exception as e:  # offline, missing package, ...
                _LIBRARY["error"] = str(e)
    return _LIBRARY["refs"] or []


def _selected_msg(refs, idx):
    return f"**Selected:** sample {idx + 1} - *\"{refs[idx].text}\"*"


def _thumb(img):
    """White margin on the right so the gallery's number badge doesn't cover the writing."""
    canvas = Image.new("L", (int(img.width * 1.12) + 8, img.height + 8), 255)
    canvas.paste(img, (4, 4))
    return canvas


def show_library():
    refs = library_refs()
    if not refs:
        return (gr.update(value=[]),
                f"Couldn't download the handwriting samples ({_LIBRARY['error']}). "
                "Use the **My own handwriting** tab instead.", "")
    items = [(_thumb(r.image), f"{i + 1}") for i, r in enumerate(refs)]
    note = (f"{len(refs)} real handwritings from the IAM database. "
            "**Click one** to use it - the highlighted one is selected.")
    return gr.update(value=items, selected_index=0), note, _selected_msg(refs, 0)


def pick_style(evt: gr.SelectData):
    refs = library_refs()
    return evt.index, _selected_msg(refs, evt.index)


# ---------------------------------------------------------------- own handwriting --------
def read_handwriting(style_files, remove_ruling, progress=gr.Progress()):
    """Find the lines in the photo and let OCR read them. The user then corrects the text."""
    paths = _paths(style_files)
    if not paths:
        raise gr.Error("Upload a photo of your handwriting first.")
    progress(0.1, desc="Finding lines")
    lines = detect_style_lines(paths, remove_ruling)
    if not lines:
        raise gr.Error("No handwriting found - try a clearer, closer photo.")
    progress(0.4, desc="Reading your handwriting (first run downloads the OCR model)")
    try:
        from copywrite.ocr import transcribe_lines

        results = transcribe_lines(lines)
    except ImportError:
        gallery = [(img, f"line {i + 1}") for i, img in enumerate(lines)]
        return gallery, "", (f"Found **{len(lines)} line(s)**. OCR isn't installed here "
                             "(pip install -r requirements-model.txt), so type what each line says below."), "own"
    gallery = [(img, f"{i + 1}. {r.text}   ({r.confidence:.0%})") for i, (img, r) in enumerate(zip(lines, results))]
    unsure = [str(i + 1) for i, r in enumerate(results) if r.needs_check]
    msg = f"Found and read **{len(lines)} line(s)**. Compare the text below with your photo and fix any mistakes."
    if unsure:
        msg += f" Check line(s) **{', '.join(unsure)}** carefully - the OCR is least sure about them."
    return gallery, "\n".join(r.text for r in results), msg, "own"


def style_refs(source, lib_idx, style_files, transcription, remove_ruling):
    if source == "library":
        refs = library_refs()
        if not refs:
            raise gr.Error("The sample handwritings aren't available - use the 'My own handwriting' tab.")
        return [refs[int(lib_idx or 0)]]
    lines = detect_style_lines(_paths(style_files), remove_ruling)
    if not lines:
        raise gr.Error("Upload a photo of your handwriting, or pick a sample handwriting instead.")
    if not transcription or not transcription.strip():
        raise gr.Error("Click 'Read my handwriting' and check the text first.")
    try:
        return build_refs(lines, transcription)
    except ValueError as e:
        raise gr.Error(str(e))




# ---------------------------------------------------------------- generate ---------------
def run(source, lib_idx, text, text_file, style_files, transcription, remove_ruling, backend, quality,
        background, bg_photo, page_size, line_spacing, text_scale, jitter, margin_left,
        ink_name, ink_custom, pen, watermark, seed, debug, progress=gr.Progress()):
    if text_file:
        text = read_text_file(_paths([text_file])[0])
    if not text or not text.strip():
        raise gr.Error("Type or paste some notes in step 2 (or upload a .txt/.docx file).")
    refs = style_refs(source, lib_idx, style_files, transcription, remove_ruling)
    if background == "photo" and bg_photo is None:
        raise gr.Error("Upload a photo of a blank page, or pick Ruled / Plain / Grid paper.")

    ink = hex_to_rgb(ink_custom) if ink_name == "custom" else INK_COLORS[ink_name]
    cfg = PageConfig(page_size=page_size, background=background, line_spacing_mm=line_spacing,
                     text_scale=text_scale, jitter=jitter, margin_left_mm=margin_left,
                     ink_color=ink, pen_weight=float(pen), watermark=watermark,
                     seed=int(seed) if seed is not None else None)

    ocr = None
    if backend == "emuru":
        progress(0.0, desc="Loading the AI model (the first run downloads ~3 GB)")
        gen = generator_for(backend)
        gen.load()
        progress(0.0, desc="Loading the handwriting reader used to check the output")
        ocr = ocr_for_checking()
    else:
        gen = generator_for(backend)

    out_dir = tempfile.mkdtemp(prefix="copywrite_")
    debug_dir = os.path.join(out_dir, "debug") if debug else None
    log(f"generate: backend={backend} quality={quality} style={source} chars={len(text)} ocr_check={ocr is not None}")

    def report(p, m):
        progress(p, desc=m)
        log(f"  {p:4.0%} {m}")

    try:
        result = generate_document(text, refs, gen, cfg, bg_photo, progress=report,
                                   ocr=ocr, quality=quality, debug_dir=debug_dir)
    except Exception as e:
        log("generation FAILED:\n" + traceback.format_exc())
        raise gr.Error(f"Generation failed: {e}")
    log("done: " + json.dumps(result["stats"]))
    files = save_pages(result["pages"], out_dir, cfg.dpi)
    debug_zip = shutil.make_archive(os.path.join(out_dir, "copywrite_debug"), "zip", debug_dir) if debug else None
    return (result["pages"], files["pdf"], gr.update(value=debug_zip, visible=bool(debug_zip)),
            json.dumps(result["stats"], indent=2),
            status_text(result["stats"], ocr is not None))


def status_text(s: dict, checked: bool) -> str:
    msg = f"Done: **{s['n_pages']} page(s)**, {s['n_lines']} lines in {s['total_seconds']:.0f} s."
    if s["backend"] == "font":
        return msg + " Download the PDF below the pages."
    if checked:
        msg += f" Every piece was read back and checked; {s.get('regenerated', 0)} were rewritten."
    else:
        msg += " (The handwriting reader isn't available, so only the width of each piece was checked.)"
    bad = s.get("unverified", 0)
    if bad:
        msg += (f"\n\n**{bad} of {s['pieces']} pieces never passed the check** and may look wrong"
                + (f" (plain-font stand-in for {s['font_fallback']}, where the model wrote nothing)"
                   if s.get("font_fallback") else "")
                + ". Try **Quality: Best**, another seed, or another handwriting.")
    if s.get("model_error"):
        msg += f"\n\nModel error seen during the run: `{s['model_error'][:200]}`"
    return msg


# ---------------------------------------------------------------- UI ---------------------
def build_ui(default_backend: str):
    engine = ("AI handwriting model (Emuru) on GPU" if default_backend == "emuru"
              else "plain font - no GPU found, so the AI model is off (run on Colab for real handwriting)")
    with gr.Blocks(title="CopyWrite AI", **({} if GRADIO_6 else {"theme": THEME, "css": CSS})) as demo:
        source = gr.State("library")
        lib_idx = gr.State(0)

        gr.Markdown(f"# CopyWrite AI\nTurn typed notes into handwritten notebook pages.\n\n"
                    f"<small>Engine: {engine}</small>", elem_id="hero")

        with gr.Row(equal_height=False):
            # ---------------- left: inputs ----------------
            with gr.Column(scale=5):
                gr.Markdown("### 1. Choose a handwriting", elem_classes="step")
                with gr.Tabs():
                    with gr.Tab("Sample handwritings") as lib_tab:
                        lib_note = gr.Markdown("Loading sample handwritings...")
                        lib_gallery = gr.Gallery(show_label=False, columns=1, height=340, allow_preview=False,
                                                 object_fit="contain", elem_id="lib-gallery")
                        lib_selected = gr.Markdown()
                    with gr.Tab("My own handwriting") as own_tab:
                        gr.Markdown("Photo of **2-5 lines** of your normal writing, taken from straight above "
                                    "in good light. Then click **Read my handwriting** and fix any wrong words.")
                        style_files = gr.File(label="Handwriting photo(s)", file_count="multiple",
                                              file_types=["image"])
                        remove_ruling = gr.Checkbox(value=True, label="Paper has printed lines (remove them)")
                        read_btn = gr.Button("Read my handwriting")
                        style_msg = gr.Markdown()
                        style_gallery = gr.Gallery(label="Lines found (OCR text, confidence)", columns=1,
                                                   height=220, object_fit="contain")
                        transcription = gr.Textbox(label="What each line says - one line of text per handwritten "
                                                         "line, must match exactly", lines=4)

                gr.Markdown("### 2. What should it write?", elem_classes="step")
                text = gr.Textbox(value=EXAMPLE_TEXT, lines=8, show_label=False,
                                  placeholder="Paste your notes here. Each new line starts a new paragraph.")
                with gr.Accordion("Or upload a .txt / .docx file", open=False):
                    text_file = gr.File(show_label=False, file_types=[".txt", ".md", ".docx"])

                gr.Markdown("### 3. Page", elem_classes="step")
                background = gr.Radio([("Ruled", "ruled"), ("Plain", "plain"), ("Grid", "grid"),
                                       ("My page photo", "photo")], value="ruled", label="Paper")
                with gr.Row():
                    ink_name = gr.Dropdown(list(INK_COLORS) + ["custom"], value="blue", label="Ink colour")
                    pen = gr.Slider(0, 2, value=0.5, step=0.25, label="Pen thickness",
                                    info="0 = fine pen, 1 = ballpoint, 2 = gel pen")
                ink_custom = gr.ColorPicker(value="#1a2d8c", label="Custom ink colour", visible=False)
                bg_photo = gr.Image(label="Photo of a blank page", type="pil", visible=False)
                with gr.Row():
                    text_scale = gr.Slider(0.7, 1.6, value=1.15, step=0.05, label="Handwriting size")
                    quality = gr.Radio([("Fast", "fast"), ("Best (slower)", "best")], value="fast",
                                       label="Quality",
                                       info="Best tries each piece several times and keeps the clearest one")
                with gr.Accordion("More options", open=False):
                    with gr.Row():
                        page_size = gr.Dropdown(list(PAGE_SIZES_MM), value="A4", label="Page size")
                        line_spacing = gr.Slider(5, 14, value=8, step=0.5, label="Line spacing (mm)")
                    with gr.Row():
                        margin_left = gr.Slider(5, 40, value=25, step=1, label="Left margin (mm)")
                        jitter = gr.Slider(0, 2, value=1, step=0.1, label="Natural messiness")
                    backend = gr.Radio([("AI handwriting model (Emuru, needs GPU)", "emuru"),
                                        ("Plain font, no AI (baseline)", "font")],
                                       value=default_backend, label="Engine")
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Random seed")
                        watermark = gr.Checkbox(value=False, label="Invisible watermark")
                        debug = gr.Checkbox(value=False, label="Save debug images (zip)")

                go = gr.Button("Write my notes", variant="primary", size="lg", elem_id="write-btn")

            # ---------------- right: result ----------------
            with gr.Column(scale=6):
                gr.Markdown("### Your pages", elem_classes="step")
                status = gr.Markdown("Pick a handwriting, paste your notes, then click **Write my notes**.",
                                     elem_id="status")
                pages = gr.Gallery(show_label=False, columns=1, height=820, object_fit="contain")
                pdf = gr.File(label="Download PDF")
                debug_zip = gr.File(label="Debug images (every piece, try and line, plus a report)", visible=False)
                with gr.Accordion("Run details", open=False):
                    stats = gr.Code(language="json", show_label=False)

        # ---------------- events ----------------
        demo.load(show_library, None, [lib_gallery, lib_note, lib_selected])
        lib_gallery.select(pick_style, None, [lib_idx, lib_selected])
        lib_tab.select(lambda: "library", None, source)
        own_tab.select(lambda: "own", None, source)
        background.change(lambda b: gr.update(visible=b == "photo"), background, bg_photo)
        ink_name.change(lambda n: gr.update(visible=n == "custom"), ink_name, ink_custom)
        read_btn.click(read_handwriting, [style_files, remove_ruling],
                       [style_gallery, transcription, style_msg, source])
        go.click(run, [source, lib_idx, text, text_file, style_files, transcription, remove_ruling, backend,
                       quality, background, bg_photo, page_size, line_spacing, text_scale, jitter, margin_left,
                       ink_name, ink_custom, pen, watermark, seed, debug], [pages, pdf, debug_zip, stats, status])
    return demo


def _preload(backend: str):
    """Load the models and the sample library while the user is still setting things up."""
    try:
        log("loading sample handwritings...")
        log(f"sample handwritings: {len(library_refs())} ({_LIBRARY['error'] or 'ok'})")
        if backend == "emuru":
            log("loading the Emuru model (first run downloads ~3 GB)...")
            generator_for("emuru").load()
            log(f"Emuru ready on {generator_for('emuru').device}")
            log("loading the TrOCR checker...")
            log("TrOCR ready" if ocr_for_checking() is not None else "TrOCR NOT available - width check only")
    except Exception:
        log("preload FAILED:\n" + traceback.format_exc())


def log(msg: str):
    """Console log (shown by the notebook's log cell)."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public link (use this on Colab/Kaggle)")
    ap.add_argument("--backend", default=os.environ.get("COPYWRITE_BACKEND"), choices=["emuru", "font"],
                    help="default: emuru if a GPU is available, otherwise font")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    backend = args.backend or ("emuru" if gpu_available() else "font")
    threading.Thread(target=_preload, args=(backend,), daemon=True).start()
    launch_kw = {"theme": THEME, "css": CSS} if GRADIO_6 else {}
    build_ui(backend).queue().launch(share=args.share, server_name=args.host, server_port=args.port, **launch_kw)
