"""End-to-end: text + style references + settings -> list of page images."""

import dataclasses
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from .background import prepare_background
from .backends import LineGenerator
from .config import PageConfig
from .layout import composite, paragraphs, place_line, plan_lines, trim_width
from .preprocess import StyleRef

CALIBRATION_TEXT = "the quick brown fox jumps over the lazy dog"

ProgressFn = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


def measure_px_per_char(generator: LineGenerator, ref: StyleRef, seed: Optional[int]) -> float:
    """Generate one known sentence and measure how wide this style writes (64-px-height units)."""
    img = trim_width(generator.generate(CALIBRATION_TEXT, ref, seed))
    return img.width / len(CALIBRATION_TEXT)


def generate_document(
    text: str,
    refs: List[StyleRef],
    generator: LineGenerator,
    cfg: PageConfig,
    background_photo: Optional[Image.Image] = None,
    progress: ProgressFn = _noop,
) -> Dict:
    if not refs:
        raise ValueError("Need at least one style reference.")
    if not text.strip():
        raise ValueError("No text to write.")
    cfg = dataclasses.replace(cfg)  # don't mutate caller's settings
    rng = random.Random(cfg.seed)
    t0 = time.time()

    progress(0.02, "Preparing page background")
    background, layout = prepare_background(cfg, background_photo)
    if not layout.slots:
        raise ValueError("No writable lines on this page - check margins / line spacing.")

    # --- calibrate text width so line wrapping is right for this handwriting ---------
    progress(0.05, "Calibrating handwriting width")
    ppc = measure_px_per_char(generator, refs[0], cfg.seed)
    line_h_px = cfg.mm(cfg.line_spacing_mm) * cfg.text_scale
    ppc_page = ppc * line_h_px / 64.0
    avail = layout.slots[0].x_end - layout.slots[0].x_start
    chars_per_line = max(8, int(avail / ppc_page * 0.95))
    indent_px = cfg.mm(cfg.paragraph_indent_mm)
    indent_chars = int(indent_px / ppc_page)

    planned = plan_lines(paragraphs(text), chars_per_line, indent_chars, cfg.blank_line_between_paragraphs)

    # --- generate every non-empty line, batched per style reference -------------------
    todo = [(i, t) for i, (t, _) in enumerate(planned) if t]
    by_ref: Dict[int, List] = {}
    for k, (i, t) in enumerate(todo):
        by_ref.setdefault(k % len(refs), []).append((i, t))  # rotate refs -> natural variation

    images: Dict[int, Image.Image] = {}
    done = 0
    for r, items in by_ref.items():
        for b in range(0, len(items), cfg.batch_size):
            chunk = items[b: b + cfg.batch_size]
            outs = generator.generate_many([t for _, t in chunk], refs[r],
                                           None if cfg.seed is None else cfg.seed + b)
            for (i, _), img in zip(chunk, outs):
                images[i] = img
            done += len(chunk)
            progress(0.1 + 0.8 * done / max(1, len(todo)), f"Generated {done}/{len(todo)} lines")
    t_gen = time.time() - t0

    # --- lay the lines out on pages --------------------------------------------------
    progress(0.92, "Composing pages")
    pages, ink = [], np.zeros((layout.height, layout.width), dtype=np.float32)
    slot_i = 0
    squeezes = []
    for i, (t, para_start) in enumerate(planned):
        if slot_i >= len(layout.slots):
            pages.append(composite(background, ink, cfg))
            ink = np.zeros_like(ink)
            slot_i = 0
        if t:
            squeezes.append(place_line(ink, images[i], layout.slots[slot_i], cfg, rng,
                                       indent_px if para_start else 0))
        slot_i += 1
    pages.append(composite(background, ink, cfg))

    if cfg.watermark:
        from .watermark import embed
        pages = [embed(p, cfg.watermark_text) for p in pages]

    progress(1.0, "Done")
    return {
        "pages": pages,
        "line_images": [images[i] for i in sorted(images)],
        "lines": [t for t, _ in planned],
        "stats": {
            "backend": generator.name,
            "n_lines": len(todo),
            "n_pages": len(pages),
            "chars_per_line": chars_per_line,
            "generation_seconds": round(t_gen, 2),
            "seconds_per_line": round(t_gen / max(1, len(todo)), 3),
            "lines_squeezed": sum(1 for s in squeezes if s < 0.999),
            "total_seconds": round(time.time() - t0, 2),
        },
    }
