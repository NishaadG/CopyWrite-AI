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
import threading
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocess import STYLE_HEIGHT, StyleRef


def blank_line() -> Image.Image:
    return Image.new("L", (16, STYLE_HEIGHT), 255)


def style_px_per_char(style: StyleRef) -> float:
    """How many px (at 64 px height) this handwriting uses per character, from its ink extent."""
    cols = np.where((np.array(style.image.convert("L")) < 160).any(axis=0))[0]
    width = (cols[-1] - cols[0] + 1) if len(cols) else style.image.width
    return max(4.0, width / max(1, len(style.text)))


class LineGenerator:
    name = "base"
    verifiable = False  # True if the output imitates the style (so width/OCR checks make sense)

    def generate(self, text: str, style: StyleRef, seed: Optional[int] = None) -> Image.Image:
        raise NotImplementedError

    def generate_many(self, texts: List[str], style: StyleRef, seed: Optional[int] = None) -> List[Image.Image]:
        return [self.generate(t, style, None if seed is None else seed + i) for i, t in enumerate(texts)]


# --------------------------------------------------------------------------------------
# Emuru (the generative model)
# --------------------------------------------------------------------------------------
class EmuruGenerator(LineGenerator):
    """Emuru writes by continuing the style image one 8-px latent column at a time, and stops
    when it has produced `stopping_after` columns that look like blank paper. The model's
    default (10 columns = 80 px) is about one wide word gap at 64 px height, so with loosely
    spaced handwriting it used to stop mid-line. We wait for 16 blank columns instead, and the
    LineWriter (writer.py) only asks for a few words at a time and checks every result."""

    name = "emuru"
    MODEL_ID = "blowing-up-groundhogs/emuru"
    STOPPING_AFTER = 16
    verifiable = True  # output imitates the style, so its width/text can be checked

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.model = None
        self._lock = threading.Lock()  # the app preloads in a background thread
        self._stop_kw = True           # pass stopping_after unless the model code rejects it
        self.last_error = None         # last swallowed model error, shown in the run report

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoModel

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch = torch
            model = AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model = model.to(self.device).eval()

    def _style_tensor(self, style: StyleRef):
        from torchvision.transforms import functional as F

        img = style.image.convert("RGB")
        if img.height != STYLE_HEIGHT:
            img = img.resize((img.width * STYLE_HEIGHT // img.height, STYLE_HEIGHT))
        t = F.normalize(F.to_tensor(img), [0.5], [0.5])
        return t.to(self.device)

    @staticmethod
    def _max_tokens(texts: List[str], style: StyleRef) -> int:
        # 1 latent token = 8 px of width. Estimate width from the style's ink px-per-character.
        px_per_char = style_px_per_char(style)
        longest = max(len(t) for t in texts)
        est = int(longest * px_per_char / 8 * 1.7) + 12 + EmuruGenerator.STOPPING_AFTER
        return int(min(max(est, 32), 384))

    @staticmethod
    def _to_line(pil) -> Image.Image:
        try:
            img = pil.convert("L")
        except Exception:  # empty crop (model stopped immediately)
            return blank_line()
        if img.width < 4:
            return blank_line()
        if np.median(np.array(img)) < 128:  # dark background -> make it ink-on-white
            img = Image.fromarray(255 - np.array(img))
        return img

    def generate(self, text: str, style: StyleRef, seed: Optional[int] = None) -> Image.Image:
        return self.generate_many([text], style, seed)[0]

    def generate_many(self, texts: List[str], style: StyleRef, seed: Optional[int] = None) -> List[Image.Image]:
        self.load()
        if seed is not None:
            self.torch.manual_seed(seed)
        s = self._style_tensor(style)
        return self._batch(texts, style, s, self._max_tokens(texts, style))

    def _batch(self, texts, style, s, max_new) -> List[Image.Image]:
        """Generate a batch; on GPU out-of-memory split it in half, on any other model error
        fall back to one-by-one so a single bad item can't lose the whole batch."""
        kw = {"max_new_tokens": max_new}
        if self._stop_kw:
            kw["stopping_after"] = self.STOPPING_AFTER
        try:
            with self.torch.inference_mode():
                outs = self.model.generate_batch(
                    style_texts=[style.text] * len(texts), gen_texts=texts,
                    style_imgs=self.torch.stack([s] * len(texts), dim=0),
                    lengths=[s.size(-1)] * len(texts), **kw)
            return [self._to_line(o) for o in outs]
        except TypeError as e:
            if self._stop_kw and "stopping_after" in str(e):  # older model code: use its default
                self._stop_kw = False
                return self._batch(texts, style, s, max_new)
            self.last_error = repr(e)
        except self.torch.cuda.OutOfMemoryError as e:
            self.torch.cuda.empty_cache()
            self.last_error = repr(e)
            if len(texts) == 1:
                return [blank_line()]
            h = len(texts) // 2
            return self._batch(texts[:h], style, s, max_new) + self._batch(texts[h:], style, s, max_new)
        except Exception as e:
            self.last_error = repr(e)
        if len(texts) > 1:  # one bad item shouldn't lose the whole batch
            return [self._batch([t], style, s, max_new)[0] for t in texts]
        try:
            with self.torch.inference_mode():
                out = self.model.generate(style_text=style.text, gen_text=texts[0], style_img=s, **kw)
            return [self._to_line(out)]
        except Exception as e:
            self.last_error = repr(e)
            return [blank_line()]


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
