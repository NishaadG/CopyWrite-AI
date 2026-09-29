"""All user-tunable settings live here, so the UI, CLI and eval script share one definition."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

PAGE_SIZES_MM = {
    "A4": (210, 297),
    "Letter": (216, 279),
    "A5": (148, 210),
}

INK_COLORS = {
    "blue": (22, 45, 140),
    "black": (25, 25, 30),
    "dark blue": (15, 30, 90),
    "red": (170, 25, 30),
}


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


@dataclass
class PageConfig:
    # Page geometry
    page_size: str = "A4"
    dpi: int = 150                      # 150 is plenty for screen/print and keeps things fast
    margin_left_mm: float = 25.0
    margin_right_mm: float = 12.0
    margin_top_mm: float = 25.0
    margin_bottom_mm: float = 15.0
    line_spacing_mm: float = 8.0        # typical Indian ruled notebook is ~7-8 mm
    paragraph_indent_mm: float = 8.0
    blank_line_between_paragraphs: bool = False

    # Background: "ruled", "plain", "grid", or "photo" (then pass a background image)
    background: str = "ruled"
    detect_lines_on_photo: bool = True  # align to ruled lines found in an uploaded photo

    # Ink / text appearance
    ink_color: Tuple[int, int, int] = INK_COLORS["blue"]
    text_scale: float = 1.15            # rendered line-image height = line_spacing * text_scale
    ink_opacity: float = 0.92

    # Natural variance (0 = perfectly straight, 1 = default, 2 = messy)
    jitter: float = 1.0
    seed: Optional[int] = 42

    # Optional invisible watermark (off by default)
    watermark: bool = False
    watermark_text: str = "HWGEN"

    # Generation
    batch_size: int = 8

    # ---- derived helpers -------------------------------------------------
    def mm(self, value_mm: float) -> int:
        return int(round(value_mm / 25.4 * self.dpi))

    @property
    def page_px(self) -> Tuple[int, int]:
        w_mm, h_mm = PAGE_SIZES_MM.get(self.page_size, PAGE_SIZES_MM["A4"])
        return self.mm(w_mm), self.mm(h_mm)


@dataclass
class LineSlot:
    """Where one handwritten line goes on a page."""
    baseline_y: int
    x_start: int
    x_end: int


@dataclass
class PageLayout:
    width: int
    height: int
    slots: list = field(default_factory=list)
