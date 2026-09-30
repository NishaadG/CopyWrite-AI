"""Layout engine: split text into lines that fit, place generated line images on the page,
add human-like variation, and blend the ink into the paper."""

import random
from typing import List, Optional, Tuple

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

    img = line.convert("L").resize((new_w, new_h), Image.LANCZOS)
    angle = rng.gauss(0, 0.35 * j)
    if abs(angle) > 0.02:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)

    ink = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
    ink = np.clip((ink - 0.08) / 0.92, 0, 1)  # drop faint model haze

    base = int(baseline * new_h / line.height) + (img.height - new_h) // 2
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
