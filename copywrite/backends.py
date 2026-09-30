"""Line generators. Each one turns (text, style reference) into a grayscale line image.

Contract for every backend:
    input : text for ONE line, a StyleRef
    output: PIL "L" image, height 64, white background, dark ink

- EmuruGenerator  : the real generative model (VAE + T5 Transformer, CVPR 2025). Needs ~3 GB
                    of weights; fast on a GPU (Colab/Kaggle T4), slow but possible on CPU.
- FontGenerator   : a handwriting *font* with random per-letter jitter. No AI. Used as the
                    baseline in evaluation (this is exactly what the problem statement says
                    is not good enough) and for developing the layout on a weak laptop.
"""

import glob
import os
import random
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocess import STYLE_HEIGHT, StyleRef


class LineGenerator:
    name = "base"

    def generate(self, text: str, style: StyleRef, seed: Optional[int] = None) -> Image.Image:
        raise NotImplementedError

    def generate_many(self, texts: List[str], style: StyleRef, seed: Optional[int] = None) -> List[Image.Image]:
        return [self.generate(t, style, None if seed is None else seed + i) for i, t in enumerate(texts)]


# --------------------------------------------------------------------------------------
# Emuru (the generative model)
# --------------------------------------------------------------------------------------
class EmuruGenerator(LineGenerator):
    name = "emuru"
    MODEL_ID = "blowing-up-groundhogs/emuru"

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.model = AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        self.model.to(self.device).eval()

    def _style_tensor(self, style: StyleRef):
        from torchvision.transforms import functional as F

        img = style.image.convert("RGB")
        if img.height != STYLE_HEIGHT:
            img = img.resize((img.width * STYLE_HEIGHT // img.height, STYLE_HEIGHT))
        t = F.normalize(F.to_tensor(img), [0.5], [0.5])
        return t.to(self.device)

    @staticmethod
    def _max_tokens(texts: List[str], style: StyleRef) -> int:
        # 1 latent token = 8 px of width. Estimate width from the style's px-per-character.
        px_per_char = style.image.width / max(len(style.text), 1)
        longest = max(len(t) for t in texts)
        est = int(longest * px_per_char / 8 * 1.5) + 16
        return int(min(max(est, 32), 448))

    def _to_line(self, pil: Image.Image) -> Image.Image:
        img = pil.convert("L")
        if np.median(np.array(img)) < 128:  # dark background -> make it ink-on-white
            img = Image.fromarray(255 - np.array(img))
        if img.width < 4:  # model failed to stop / produced nothing
            img = Image.new("L", (16, STYLE_HEIGHT), 255)
        return img

    def generate(self, text: str, style: StyleRef, seed: Optional[int] = None) -> Image.Image:
        return self.generate_many([text], style, seed)[0]

    def generate_many(self, texts: List[str], style: StyleRef, seed: Optional[int] = None) -> List[Image.Image]:
        self.load()
        if seed is not None:
            self.torch.manual_seed(seed)
        s = self._style_tensor(style)
        max_new = self._max_tokens(texts, style)
        if len(texts) == 1:
            out = self.model.generate(style_text=style.text, gen_text=texts[0], style_img=s, max_new_tokens=max_new)
            return [self._to_line(out)]
        outs = self.model.generate_batch(
            style_texts=[style.text] * len(texts),
            gen_texts=texts,
            style_imgs=self.torch.stack([s] * len(texts), dim=0),
            lengths=[s.size(-1)] * len(texts),
            max_new_tokens=max_new,
        )
        return [self._to_line(o) for o in outs]


# --------------------------------------------------------------------------------------
# Font baseline (no AI)
# --------------------------------------------------------------------------------------
def find_font(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = sorted(glob.glob(os.path.join(here, "assets", "fonts", "*.ttf")))
    candidates += [
        "C:/Windows/Fonts/segoepr.ttf",   # Segoe Print (Windows)
        "C:/Windows/Fonts/segoesc.ttf",   # Segoe Script (Windows)
        "C:/Windows/Fonts/Inkfree.ttf",   # Ink Free (Windows)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class FontGenerator(LineGenerator):
    name = "font"

    def __init__(self, font_path: Optional[str] = None, jitter: float = 1.0):
        self.font_path = find_font(font_path)
        self.jitter = jitter
        self._fonts = {}

    def _font(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = (
                ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default()
            )
        return self._fonts[size]

    def generate(self, text: str, style: StyleRef, seed: Optional[int] = None) -> Image.Image:
        rng = random.Random(seed)
        j = self.jitter
        base_size = 38
        baseline = 46
        x = 6
        canvas = Image.new("L", (max(64, int(len(text) * base_size * 0.9) + 40), STYLE_HEIGHT), 255)
        for ch in text:
            size = max(10, int(base_size * (1 + rng.gauss(0, 0.04 * j))))
            font = self._font(size)
            if ch == " ":
                x += int(size * (0.33 + rng.uniform(-0.05, 0.08) * j))
                continue
            bbox = font.getbbox(ch)
            w = max(1, bbox[2] - bbox[0])
            glyph = Image.new("L", (w + 20, STYLE_HEIGHT + 20), 0)
            ascent = font.getmetrics()[0]
            ImageDraw.Draw(glyph).text((10 - bbox[0], 10 + baseline - ascent), ch, font=font, fill=255)
            glyph = glyph.rotate(rng.gauss(0, 3.0 * j), resample=Image.BICUBIC, center=(10 + w / 2, 10 + baseline))
            dy = int(round(rng.gauss(0, 0.8 * j)))
            if x + w + 20 >= canvas.width:
                break
            canvas.paste(0, (x - 10, -10 + dy), glyph)
            x += w + int(rng.gauss(1.0, 0.8 * j))
        return canvas.crop((0, 0, min(canvas.width, x + 8), STYLE_HEIGHT))


def get_generator(name: str, **kwargs) -> LineGenerator:
    name = name.lower()
    if name.startswith("emuru"):
        return EmuruGenerator(**kwargs)
    if name.startswith("font"):
        return FontGenerator(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Use 'emuru' or 'font'.")
