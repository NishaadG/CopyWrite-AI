"""End-to-end: text + style references + settings -> list of page images.

Stages:
  1. background   page template or photo, and the line slots to write on
  2. width model  how many characters fit on a line in this handwriting
  3. wrap         text -> lines
  4. write        LineWriter: pieces -> batched generation -> verification -> retries -> stitch
  5. layout       scale, align baselines to the rules, jitter, blend ink into the paper
"""

import dataclasses
import os
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from .background import prepare_background
from .backends import LineGenerator, style_px_per_char
from .config import PageConfig
from .layout import composite, paragraphs, place_line, plan_lines, trim_width
from .preprocess import STYLE_HEIGHT, StyleRef
from .writer import LineWriter, WriterConfig

CALIBRATION_TEXT = "the quick brown fox jumps over the lazy dog"

ProgressFn = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


def px_per_char(generator: LineGenerator, refs: List[StyleRef], seed: Optional[int]) -> float:
    """Width of this handwriting in px per character, at 64 px line height.

    A style-imitating model writes about as wide as the style sample, so we measure the sample
    itself (no model call, and it can't be fooled by a truncated calibration line). The font
    baseline ignores the style, so for it we measure one rendered sentence.
    """
    if getattr(generator, "verifiable", False):
        return float(np.median([style_px_per_char(r) for r in refs]))
    img = trim_width(generator.generate(CALIBRATION_TEXT, refs[0], seed))
    return img.width / len(CALIBRATION_TEXT)


def generate_document(
    text: str,
    refs: List[StyleRef],
    generator: LineGenerator,
    cfg: PageConfig,
    background_photo: Optional[Image.Image] = None,
    progress: ProgressFn = _noop,
    ocr=None,
    quality: str = "fast",
    debug_dir: Optional[str] = None,
) -> Dict:
    """`ocr`: a HandwritingOCR used to check generated text (optional but recommended with emuru).
    `debug_dir`: if set, every intermediate image and a per-piece report are saved there."""
    if not refs:
        raise ValueError("Need at least one style reference.")
    if not text.strip():
        raise ValueError("No text to write.")
    cfg = dataclasses.replace(cfg)  # don't mutate caller's settings
    rng = random.Random(cfg.seed)
    t0 = time.time()

    progress(0.01, "Preparing the page")
    background, layout = prepare_background(cfg, background_photo)
    if not layout.slots:
        raise ValueError("No writable lines on this page - check margins / line spacing.")

    ppc = px_per_char(generator, refs, cfg.seed)
    line_h_px = cfg.mm(cfg.line_spacing_mm) * cfg.text_scale
    ppc_page = ppc * line_h_px / STYLE_HEIGHT
    avail = layout.slots[0].x_end - layout.slots[0].x_start
    chars_per_line = max(8, int(avail / ppc_page * 0.92))
    indent_px = cfg.mm(cfg.paragraph_indent_mm)
    indent_chars = int(indent_px / ppc_page)
    planned = plan_lines(paragraphs(text), chars_per_line, indent_chars, cfg.blank_line_between_paragraphs)

    writer = LineWriter(generator, refs, WriterConfig.for_quality(quality, batch_size=cfg.batch_size),
                        ocr=ocr, seed=cfg.seed, debug_dir=debug_dir,
                        progress=lambda p, m: progress(0.03 + 0.9 * p, m))
    written = writer.write([t for t, _ in planned])
    t_gen = time.time() - t0

    progress(0.95, "Laying out the pages")
    pages, ink = [], np.zeros((layout.height, layout.width), dtype=np.float32)
    slot_i = 0
    squeezes = []
    for i, (t, para_start) in enumerate(planned):
        if slot_i >= len(layout.slots):
            pages.append(composite(background, ink, cfg))
            ink = np.zeros_like(ink)
            slot_i = 0
        if t:
            w = written[i]
            squeezes.append(place_line(ink, w.image, layout.slots[slot_i], cfg, rng,
                                       indent_px if para_start else 0, baseline=w.baseline))
        slot_i += 1
    pages.append(composite(background, ink, cfg))

    if cfg.watermark:
        from .watermark import embed
        pages = [embed(p, cfg.watermark_text) for p in pages]

    if debug_dir:
        for k, p in enumerate(pages, 1):
            p.save(os.path.join(debug_dir, f"page_{k:02d}.png"))

    n_lines = sum(1 for t, _ in planned if t)
    progress(1.0, "Done")
    return {
        "pages": pages,
        "line_images": [written[i].image for i in sorted(written)],
        "lines": [t for t, _ in planned],
        "stats": {
            "backend": generator.name,
            "quality": quality,
            "ocr_check": ocr is not None and getattr(generator, "verifiable", False),
            "n_lines": n_lines,
            "n_pages": len(pages),
            "chars_per_line": chars_per_line,
            **writer.report.as_dict(),
            "generation_seconds": round(t_gen, 2),
            "seconds_per_line": round(t_gen / max(1, n_lines), 3),
            "lines_squeezed": sum(1 for s in squeezes if s < 0.999),
            "total_seconds": round(time.time() - t0, 2),
        },
    }
