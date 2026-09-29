"""Command-line entry point (useful in Colab/Kaggle, for batch runs and for evaluation).

Example:
    python -m hwgen.cli --text samples/text/sample_notes.txt \
        --style samples/style/my_lines.jpg --style-text samples/style/my_lines.txt \
        --backend emuru --background ruled --out outputs/run1
"""

import argparse
import json

from PIL import Image

from .backends import get_generator
from .config import INK_COLORS, PageConfig, hex_to_rgb
from .io_utils import read_text_file, save_pages
from .pipeline import generate_document
from .preprocess import prepare_style_refs


def main():
    ap = argparse.ArgumentParser(description="Generate handwritten notes in your handwriting.")
    ap.add_argument("--text", required=True, help=".txt / .md / .docx file with the content")
    ap.add_argument("--style", nargs="+", required=True, help="photo(s) of your handwriting")
    ap.add_argument("--style-text", required=True,
                    help="file (or literal string) with the transcription, one line per handwritten line")
    ap.add_argument("--backend", default="emuru", choices=["emuru", "font"])
    ap.add_argument("--background", default="ruled", choices=["ruled", "plain", "grid", "photo"])
    ap.add_argument("--background-photo", default=None)
    ap.add_argument("--ink", default="blue", help=f"{list(INK_COLORS)} or hex like #1a2b8c")
    ap.add_argument("--line-spacing-mm", type=float, default=8.0)
    ap.add_argument("--text-scale", type=float, default=1.15)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watermark", action="store_true", help="embed an invisible watermark (optional)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--keep-ruling", action="store_true", help="don't remove notebook lines from style photo")
    ap.add_argument("--out", default="outputs/run")
    args = ap.parse_args()

    try:
        style_text = read_text_file(args.style_text)
    except (FileNotFoundError, OSError):
        style_text = args.style_text

    refs = prepare_style_refs(args.style, style_text, remove_lines=not args.keep_ruling)
    ink = INK_COLORS.get(args.ink) or hex_to_rgb(args.ink)
    cfg = PageConfig(background=args.background, ink_color=ink, line_spacing_mm=args.line_spacing_mm,
                     text_scale=args.text_scale, jitter=args.jitter, seed=args.seed,
                     watermark=args.watermark, batch_size=args.batch_size)
    photo = Image.open(args.background_photo) if args.background_photo else None

    def progress(p, msg):
        print(f"[{p * 100:5.1f}%] {msg}", flush=True)

    result = generate_document(read_text_file(args.text), refs, get_generator(args.backend), cfg, photo, progress)
    files = save_pages(result["pages"], args.out, cfg.dpi)
    for i, r in enumerate(refs):
        r.image.save(f"{args.out}/style_ref_{i}.png")
    print(json.dumps({**result["stats"], **files}, indent=2))


if __name__ == "__main__":
    main()
