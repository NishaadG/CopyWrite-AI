"""hwgen - handwritten notes generator.

Few-shot styled handwriting generation (Emuru: VAE + T5 Transformer) + a classical
layout engine that writes the result onto ruled / plain / photographed pages.
"""

from .backends import EmuruGenerator, FontGenerator, get_generator
from .config import INK_COLORS, PageConfig
from .pipeline import generate_document
from .preprocess import StyleRef, prepare_style_refs, single_ref

__all__ = [
    "EmuruGenerator", "FontGenerator", "get_generator", "INK_COLORS", "PageConfig",
    "generate_document", "StyleRef", "prepare_style_refs", "single_ref",
]
