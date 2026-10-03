"""Layout engine: split text into lines that fit, place generated line images on the page,
add human-like variation, and blend the ink into the paper."""

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import LineSlot, PageConfig

WORD_SEP = " "

# Word/.docx typography -> plain characters the handwriting model has seen (it reads raw bytes,
# so a curly quote is 3 unknown bytes to it)
_PLAIN = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
          "…": "...", "•": "-", " ": " ", "\t": " ", "×": "x"}


def normalize_text(text: str) -> str:
    return "".join(_PLAIN.get(ch, ch) for ch in text)


def paragraphs(text: str) -> List[List[str]]:
    """Each non-empty input line is treated as a paragraph (keeps the user's own line breaks)."""
    out = []
    for raw in normalize_text(text).replace("\r\n", "\n").split("\n"):
        words = raw.split()
        out.append(words)  # empty list = blank line, preserved
    while out and not out[-1]:
        out.pop()
    return out


def plan_lines(paras: List[List[str]], chars_per_line: int, indent_chars: int, blank_between: bool) -> List[Tuple[str, bool]]:
    """Greedy word wrap by estimated character capacity.

    Returns (line_text, is_paragraph_start). Blank input lines become ("", False).
    """
    lines: List[Tuple[str, bool]] = []
    for pi, words in enumerate(paras):
        if not words:
            lines.append(("", False))
            continue
        cur, first = [], True
        cap = chars_per_line - indent_chars
        for w in words:
            candidate = len(WORD_SEP.join(cur + [w]))
            if cur and candidate > cap:
                lines.append((WORD_SEP.join(cur), first))
                cur, first, cap = [w], False, chars_per_line
            else:
                cur.append(w)
        if cur:
            lines.append((WORD_SEP.join(cur), first))
        if blank_between and pi < len(paras) - 1:
            lines.append(("", False))
    return lines


def estimate_baseline(line: Image.Image) -> int:
    """Row of the writing baseline: bottom of the densest band of ink (ignores descenders)."""
    arr = 255 - np.array(line.convert("L"), dtype=np.float32)
    rows = arr.sum(axis=1)
    if rows.max() <= 0:
        return int(line.height * 0.72)
    rows = np.convolve(rows, np.ones(3) / 3, mode="same")
    dense = np.where(rows > rows.max() * 0.35)[0]
    return int(dense[-1]) if len(dense) else int(line.height * 0.72)


def trim_width(line: Image.Image, pad: int = 2) -> Image.Image:
    arr = np.array(line.convert("L"))
    cols = np.where((arr < 160).any(axis=0))[0]
    if len(cols) == 0:
        return line
    return line.crop((max(0, cols[0] - pad), 0, min(line.width, cols[-1] + pad + 1), line.height))


def ink_alpha(line: Image.Image, pen_weight: float = 0.5) -> np.ndarray:
    """Line image -> ink coverage (0 = paper, 1 = full ink), with full-strength strokes.

    Emuru's VAE decoder draws strokes in dark grey, not black, with a faint haze around
    them, and thin grey strokes look washed out once scaled onto the page. So per line:
    the paper/haze level maps to 0, the strongest strokes (90th percentile of stroke
    pixels) map to 1. Lines that are already black barely change. `pen_weight` (0-2)
    then thickens strokes by up to ~1 px per side per unit at 64 px height, like a bolder pen.
    """
    a = 1.0 - np.asarray(line.convert("L"), dtype=np.float32) / 255.0
    strokes = a[a > 0.15]
    if strokes.size < 10:
        return np.clip(a, 0, 1)
    lo = max(0.10, float(np.median(a)) + 0.06)                # paper + haze
    hi = max(lo + 0.15, float(np.percentile(strokes, 90)))    # what counts as full ink
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    w = max(0.0, pen_weight)
    while w > 0:  # each unit = one 3x3 dilation; fractions blend
        a = a + (cv2.dilate(a, kernel) - a) * min(1.0, w)
        w -= 1.0
    return a ** 0.8  # solid stroke cores, soft (anti-aliased) edges


def _rotate(a: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an ink map around its centre, expanding the canvas so nothing is cut off."""
    h, w = a.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(a, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


def place_line(
    page_ink: np.ndarray,
    line: Image.Image,
    slot: LineSlot,
    cfg: PageConfig,
    rng: random.Random,
    indent_px: int = 0,
    baseline: Optional[int] = None,
) -> float:
    """Scale, jitter and write one line into the page's ink-alpha buffer (max-combine).

    `baseline` is the row of the writing baseline in `line` (estimated if not given).
    Returns the horizontal squeeze factor used to fit the line (1.0 = none; for diagnostics).
    """
    j = cfg.jitter
    if baseline is None:
        baseline = estimate_baseline(line)
    line = trim_width(line)
    line_h_px = cfg.mm(cfg.line_spacing_mm) * cfg.text_scale
    scale = line_h_px / line.height * (1 + rng.gauss(0, 0.015 * j))
    new_w = max(1, int(line.width * scale))
    new_h = max(1, int(line.height * scale))

    avail = slot.x_end - (slot.x_start + indent_px)
    squeeze = 1.0
    if new_w > avail:  # overflow: squeeze horizontally a bit, then shrink uniformly
        squeeze = avail / new_w
        if squeeze < 0.85:
            shrink = squeeze / 0.85
            new_h = max(1, int(new_h * shrink))
            scale *= shrink
            squeeze = 0.85
        new_w = avail

    ink = ink_alpha(line, cfg.pen_weight)  # at the model's resolution, before any resizing
    interp = cv2.INTER_AREA if new_h < line.height else cv2.INTER_CUBIC
    ink = np.clip(cv2.resize(ink, (new_w, new_h), interpolation=interp), 0, 1)
    angle = rng.gauss(0, 0.35 * j)
    if abs(angle) > 0.02:
        ink = _rotate(ink, angle)

    base = int(baseline * new_h / line.height) + (ink.shape[0] - new_h) // 2
    x = slot.x_start + indent_px + int(abs(rng.gauss(0, 3.0 * j)))
    y = slot.baseline_y - base - 2 + int(round(rng.gauss(0, 1.2 * j)))

    H, W = page_ink.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + ink.shape[1]), min(H, y + ink.shape[0])
    if x1 > x0 and y1 > y0:
        sub = ink[y0 - y: y1 - y, x0 - x: x1 - x]
        page_ink[y0:y1, x0:x1] = np.maximum(page_ink[y0:y1, x0:x1], sub)
    return squeeze


def composite(background: Image.Image, ink_alpha: np.ndarray, cfg: PageConfig) -> Image.Image:
    """Multiply-blend coloured ink onto the paper so paper texture/lighting shows through."""
    bg = np.asarray(background.convert("RGB"), dtype=np.float32) / 255.0
    ink_rgb = np.array(cfg.ink_color, dtype=np.float32) / 255.0
    a = (ink_alpha * cfg.ink_opacity)[..., None]
    # multiply: ink darkens paper proportionally; paper's own shading is preserved
    out = bg * (1 - a) + (bg * ink_rgb) * a
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))
