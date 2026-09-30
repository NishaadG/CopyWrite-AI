"""Page backgrounds: generated templates (ruled / plain / grid) or a user's photo,
plus classical-CV detection of the ruled lines in that photo so text sits on them."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import LineSlot, PageConfig, PageLayout

PAPER = (250, 249, 244)
RULE_BLUE = (150, 180, 215)
MARGIN_RED = (215, 120, 120)


def make_template(cfg: PageConfig, kind: str) -> Image.Image:
    w, h = cfg.page_px
    page = Image.new("RGB", (w, h), PAPER)
    arr = np.array(page)
    # faint paper grain so it doesn't look like a flat digital fill
    rng = np.random.default_rng(0)
    arr = np.clip(arr.astype(np.int16) + rng.normal(0, 2.0, arr.shape[:2])[..., None], 0, 255).astype(np.uint8)
    step = cfg.mm(cfg.line_spacing_mm)
    top = cfg.mm(cfg.margin_top_mm)
    if kind == "ruled":
        for y in range(top, h - cfg.mm(5), step):
            arr[y:y + 2, :] = RULE_BLUE
        x = cfg.mm(cfg.margin_left_mm) - cfg.mm(4)
        arr[:, x:x + 2] = MARGIN_RED
    elif kind == "grid":
        grid = cfg.mm(5)
        arr[::grid, :] = (205, 215, 225)
        arr[:, ::grid] = (205, 215, 225)
    return Image.fromarray(arr)


def fit_photo(photo: Image.Image, cfg: PageConfig) -> Image.Image:
    """Resize the user's page photo to the configured page size (cover + centre crop)."""
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    return ImageOps.fit(photo, cfg.page_px, Image.LANCZOS)


def detect_ruled_lines(page: Image.Image) -> Tuple[List[int], Optional[int]]:
    """Return (y positions of horizontal ruled lines, x of the vertical margin line or None)."""
    gray = cv2.cvtColor(np.array(page.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8)

    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1)))
    rows = horiz.sum(axis=1) / 255.0
    ys = _peaks(rows, min_value=w * 0.25, min_dist=max(8, h // 120))
    ys = _keep_regular(ys)

    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3)))
    cols = vert.sum(axis=0) / 255.0
    xs = [x for x in _peaks(cols, min_value=h * 0.4, min_dist=10) if x < w * 0.3]
    margin_x = xs[-1] if xs else None
    return ys, margin_x


def _peaks(signal: np.ndarray, min_value: float, min_dist: int) -> List[int]:
    idx = [i for i in range(1, len(signal) - 1)
           if signal[i] >= min_value and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]]
    out = []
    for i in idx:
        if out and i - out[-1] < min_dist:
            if signal[i] > signal[out[-1]]:
                out[-1] = i
        else:
            out.append(i)
    return out


def _keep_regular(ys: List[int]) -> List[int]:
    """Ruled lines are evenly spaced; drop detections that break the rhythm (headers, stray marks)."""
    if len(ys) < 4:
        return ys
    gaps = np.diff(ys)
    med = float(np.median(gaps))
    keep = [ys[0]]
    for y in ys[1:]:
        if abs((y - keep[-1]) - med) <= med * 0.3:
            keep.append(y)
        elif (y - keep[-1]) > med * 1.3:
            keep.append(y)  # missed a line in between; still fine
    return keep


def build_layout(cfg: PageConfig, page: Image.Image, ruled_ys: Optional[List[int]] = None,
                 margin_x: Optional[int] = None) -> PageLayout:
    w, h = page.size
    x0 = (margin_x + cfg.mm(3)) if margin_x else cfg.mm(cfg.margin_left_mm)
    x1 = w - cfg.mm(cfg.margin_right_mm)
    if ruled_ys and len(ruled_ys) >= 3:
        # first ruled line is usually the header line -> start writing on the next one
        usable = [y for y in ruled_ys if y > cfg.mm(cfg.margin_top_mm) * 0.8 and y < h - cfg.mm(cfg.margin_bottom_mm) * 0.5]
        ys = usable[1:] if len(usable) > 3 else usable
    else:
        step = cfg.mm(cfg.line_spacing_mm)
        ys = list(range(cfg.mm(cfg.margin_top_mm) + step, h - cfg.mm(cfg.margin_bottom_mm), step))
    return PageLayout(width=w, height=h, slots=[LineSlot(baseline_y=y, x_start=x0, x_end=x1) for y in ys])


def prepare_background(cfg: PageConfig, photo: Optional[Image.Image] = None) -> Tuple[Image.Image, PageLayout]:
    if cfg.background == "photo" and photo is not None:
        page = fit_photo(photo, cfg)
        if cfg.detect_lines_on_photo:
            ys, mx = detect_ruled_lines(page)
            if len(ys) >= 3:
                # adopt the paper's own line spacing so text size matches
                cfg.line_spacing_mm = float(np.median(np.diff(ys))) * 25.4 / cfg.dpi
                return page, build_layout(cfg, page, ys, mx)
        return page, build_layout(cfg, page)
    kind = cfg.background if cfg.background in ("ruled", "plain", "grid") else "ruled"
    page = make_template(cfg, kind)
    if kind == "ruled":
        # text sits on the printed rules; the template rules start at margin_top
        step = cfg.mm(cfg.line_spacing_mm)
        ys = list(range(cfg.mm(cfg.margin_top_mm), page.height - cfg.mm(5), step))
        return page, build_layout(cfg, page, ys, cfg.mm(cfg.margin_left_mm) - cfg.mm(4))
    return page, build_layout(cfg, page)
