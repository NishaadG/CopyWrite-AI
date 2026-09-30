"""Turn a phone photo of someone's handwriting into clean, model-ready style references.

Pipeline: grayscale -> flatten uneven lighting -> remove notebook ruling -> split into
text lines -> crop tightly -> resize to 64 px height (what the Emuru model expects).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

STYLE_HEIGHT = 64
MAX_STYLE_WIDTH = 768  # longer references slow generation a lot (no KV-cache in the model)


@dataclass
class StyleRef:
    image: Image.Image  # grayscale "L", white background, dark ink, height 64
    text: str           # exact transcription of what the image says


def _to_gray(img: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = ImageOps.exif_transpose(img).convert("L")
        return np.array(img)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a blurred background estimate so shadows / uneven light disappear."""
    k = max(15, (min(gray.shape) // 8) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    background = cv2.GaussianBlur(background, (k, k), 0)
    flat = cv2.divide(gray, background, scale=255)
    # contrast stretch
    lo, hi = np.percentile(flat, 1), np.percentile(flat, 99.5)
    flat = np.clip((flat.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)
    return flat.astype(np.uint8)


def remove_ruling(gray: np.ndarray) -> np.ndarray:
    """Erase long horizontal/vertical printed lines (ruled notebook paper) and keep the ink."""
    inv = 255 - gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 3))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    if lines.sum() == 0:
        return gray
    out = gray.copy()
    out[lines > 0] = 255
    # handwriting strokes that crossed a ruled line now have a tiny gap; close it lightly
    ink = 255 - out
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    return 255 - ink


def ink_mask(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw > 0


def split_lines(gray: np.ndarray, min_gap: int = 3) -> List[np.ndarray]:
    """Split a multi-line image into single text lines using a horizontal projection profile."""
    mask = ink_mask(gray)
    rows = mask.sum(axis=1).astype(np.float32)
    if rows.max() == 0:
        return []
    rows = np.convolve(rows, np.ones(5) / 5, mode="same")
    is_text = rows > rows.max() * 0.04
    bands, start = [], None
    for y, t in enumerate(is_text):
        if t and start is None:
            start = y
        elif not t and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, len(is_text)])
    # merge bands separated by tiny gaps (dots of i/j, accents)
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < min_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    heights = [b[1] - b[0] for b in merged]
    if not heights:
        return []
    med = float(np.median(heights))
    # drop specks, then pad each band a little so descenders/ascenders survive
    out = []
    for y0, y1 in merged:
        if y1 - y0 < med * 0.35:
            continue
        pad = int(med * 0.25)
        out.append(gray[max(0, y0 - pad): min(gray.shape[0], y1 + pad)])
    return out


def crop_to_ink(gray: np.ndarray, pad: int = 6) -> np.ndarray:
    mask = ink_mask(gray)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return gray
    y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad + 1)
    return gray[y0:y1, x0:x1]


def to_style_height(gray: np.ndarray, height: int = STYLE_HEIGHT) -> Image.Image:
    h, w = gray.shape
    new_w = max(8, int(round(w * height / h)))
    return Image.fromarray(gray).resize((new_w, height), Image.LANCZOS)


def detect_style_lines(images: List[Union[str, Image.Image]], remove_lines: bool = True) -> List[Image.Image]:
    """Clean the photo(s) and return each handwritten line as a tightly cropped,
    full-resolution grayscale image (reading order, across all photos)."""
    crops: List[Image.Image] = []
    for img in images:
        gray = flatten_illumination(_to_gray(img))
        if remove_lines:
            gray = remove_ruling(gray)
        crops.extend(Image.fromarray(crop_to_ink(c)) for c in split_lines(gray))
    return crops


def build_refs(line_images: List[Image.Image], transcription: Union[str, List[str]]) -> List[StyleRef]:
    """Pair detected line images with their text (one text line per image)."""
    if isinstance(transcription, str):
        wanted = [t.strip() for t in transcription.strip().splitlines() if t.strip()]
    else:
        wanted = [t.strip() for t in transcription if t and t.strip()]
    if len(line_images) != len(wanted):
        raise ValueError(
            f"Found {len(line_images)} handwritten line(s) in the photo(s) but {len(wanted)} line(s) of "
            f"transcription. Keep exactly one transcription line per handwritten line "
            f"(or crop the photo to fewer lines)."
        )
    refs = []
    for crop, text in zip(line_images, wanted):
        img = to_style_height(np.array(crop.convert("L")))
        if img.width > MAX_STYLE_WIDTH:
            img, text = shorten_ref(img, text, MAX_STYLE_WIDTH)
        refs.append(StyleRef(image=img, text=text))
    return refs


def prepare_style_refs(
    images: List[Union[str, Image.Image]],
    transcription: Optional[str] = None,
    remove_lines: bool = True,
    ocr_model: Optional[str] = None,
) -> List[StyleRef]:
    """Build StyleRefs from one or more photos.

    `transcription` has one line of text per handwritten line, in reading order, across all
    images. If it is empty/None, the lines are read automatically with handwriting OCR.
    """
    lines = detect_style_lines(images, remove_lines)
    if not lines:
        raise ValueError("No handwriting found in the photo(s).")
    if not transcription or not transcription.strip():
        from .ocr import DEFAULT_MODEL, transcribe_lines

        transcription = [r.text for r in transcribe_lines(lines, ocr_model or DEFAULT_MODEL)]
    return build_refs(lines, transcription)


def word_gaps(img: Image.Image) -> List[int]:
    """x-centres of the gaps between words (empty column runs wider than ~1/4 line height)."""
    cols = ink_mask(np.array(img)).sum(axis=0)
    min_run = max(6, img.height // 4)
    gaps, run_start = [], None
    for x, c in enumerate(cols):
        if c == 0 and run_start is None:
            run_start = x
        elif c != 0 and run_start is not None:
            if x - run_start >= min_run and run_start > 0:
                gaps.append((run_start + x) // 2)
            run_start = None
    return gaps


def shorten_ref(img: Image.Image, text: str, max_width: int):
    """Cut a long reference at a word gap and keep the matching number of words.

    Only cuts when the number of detected gaps matches the number of words, so the
    transcription stays correct; otherwise returns the reference unchanged.
    """
    words = text.split()
    gaps = word_gaps(img)
    if len(gaps) != len(words) - 1:
        return img, text  # can't cut safely; slower but correct
    usable = [i for i, g in enumerate(gaps) if g <= max_width]
    if not usable:
        return img, text
    i = usable[-1]
    return img.crop((0, 0, gaps[i], img.height)), " ".join(words[: i + 1])


def single_ref(image: Union[str, Image.Image], text: str, remove_lines: bool = True) -> StyleRef:
    """Convenience for a pre-cropped single-line image."""
    gray = flatten_illumination(_to_gray(image))
    if remove_lines:
        gray = remove_ruling(gray)
    return StyleRef(image=to_style_height(crop_to_ink(gray)), text=text.strip())
