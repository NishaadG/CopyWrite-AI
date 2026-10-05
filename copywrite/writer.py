"""LineWriter: turns text lines into handwritten line images reliably.

A generative model can fail on any single call: stop early (text missing), run on (scribbles),
or misspell. So we never trust one call with a whole line:

  1. split     each line into short pieces of a few whole words (Emuru is much more reliable
               on short text, and short pieces batch well on the GPU)
  2. generate  all pieces in batches, per style reference
  3. verify    every piece: its ink width must fit the text length, and (if OCR is available)
               TrOCR must read back roughly the right letters
  4. retry     failed pieces with new random seeds; if a multi-word piece keeps failing, split
               it into single words and try again; keep the best attempt seen
  5. stitch    the pieces of a line side by side, using the handwriting's own word spacing

Nothing is dropped silently: a piece that never passes keeps its best attempt (or, if the model
produced nothing at all, a plain font rendering) and is counted in the report.
"""

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from .backends import FontGenerator, LineGenerator, style_px_per_char
from .layout import estimate_baseline, trim_width
from .metrics import letters, loose_cer, missing_letters
from .preprocess import STYLE_HEIGHT, StyleRef, ink_mask

QUALITY = {  # (candidates on the first pass, retry rounds)
    "fast": (1, 3),
    "best": (3, 4),
}


@dataclass
class WriterConfig:
    max_piece_chars: int = 28      # ~4-5 words per model call
    candidates: int = 1            # attempts per piece on the first pass (best-of-N)
    retries: int = 3               # extra rounds for pieces that fail verification
    min_width_ratio: float = 0.7   # ink width / expected width outside this range = failed
    max_width_ratio: float = 1.9
    max_cer: float = 0.25          # loose CER of the OCR read-back
    max_missing: float = 0.05      # share of letters that may be missing (OCR noise): 0 under 20 letters
    batch_size: int = 12

    @classmethod
    def for_quality(cls, quality: str = "fast", **kw) -> "WriterConfig":
        n, r = QUALITY.get(quality, QUALITY["fast"])
        return cls(candidates=n, retries=r, **kw)


@dataclass
class Piece:
    line: int
    text: str
    ref: int
    image: Optional[Image.Image] = None
    score: float = float("inf")
    ok: bool = False
    attempts: int = 0
    ocr: str = ""
    cer: Optional[float] = None
    width_ratio: float = 0.0
    missing: Optional[int] = None  # letters of the text not found in the OCR read-back
    fallback: bool = False


@dataclass
class WrittenLine:
    image: Image.Image
    baseline: int  # row of the writing baseline in `image`


@dataclass
class WriteReport:
    pieces: int = 0
    regenerated: int = 0
    split: int = 0
    unverified: int = 0
    font_fallback: int = 0
    model_calls: int = 0
    model_error: Optional[str] = None
    rows: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "rows"}
        return {k: v for k, v in d.items() if v is not None}


def split_pieces(text: str, max_chars: int) -> List[str]:
    """Greedy split into runs of whole words, each at most max_chars (a longer word stays alone)."""
    pieces, cur = [], []
    for w in text.split():
        if cur and len(" ".join(cur + [w])) > max_chars:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def pick_ref(text: str, refs: List[StyleRef], rotation: int) -> int:
    """Style line sharing the most distinct characters with the text; ties rotate for variety."""
    if len(refs) == 1:
        return 0
    want = set(text.lower())
    scores = [len(want & set(r.text.lower())) for r in refs]
    best = max(scores)
    tied = [i for i, s in enumerate(scores) if s == best]
    return tied[rotation % len(tied)]


def word_gap_px(img: Image.Image) -> int:
    """Typical gap between words in a handwriting image (median blank-column run)."""
    cols = ink_mask(np.array(img.convert("L"))).any(axis=0)
    runs, run = [], 0
    for has_ink in cols:
        if has_ink:
            if run:
                runs.append(run)
            run = 0
        else:
            run += 1
    gaps = [r for r in runs if r >= max(6, img.height // 6)]  # ignore gaps inside words
    return int(np.clip(np.median(gaps), 12, 48)) if gaps else 24


def stitch(images: List[Image.Image], gap: int) -> Image.Image:
    parts = [trim_width(im, pad=0) for im in images]
    h = max(p.height for p in parts)
    w = sum(p.width for p in parts) + gap * (len(parts) - 1) + 8
    out = Image.new("L", (w, h), 255)
    x = 4
    for p in parts:
        out.paste(p.convert("L"), (x, 0))
        x += p.width + gap
    return out


class LineWriter:
    def __init__(self, generator: LineGenerator, refs: List[StyleRef], wcfg: Optional[WriterConfig] = None,
                 ocr=None, seed: Optional[int] = None, debug_dir: Optional[str] = None,
                 progress: Callable[[float, str], None] = lambda p, m: None):
        self.gen, self.refs = generator, refs
        self.verify = getattr(generator, "verifiable", False)
        self.wcfg = wcfg or WriterConfig()
        if not self.verify:  # e.g. the font baseline: one call per line, nothing to check
            self.wcfg = WriterConfig(max_piece_chars=10 ** 6, candidates=1, retries=0,
                                     batch_size=self.wcfg.batch_size)
        self.ocr = ocr if self.verify else None
        self.seed = seed if seed is not None else 0
        self.debug_dir = debug_dir
        self.progress = progress
        self.ppc = [style_px_per_char(r) for r in refs]
        self.gaps = [word_gap_px(r.image) for r in refs]
        self.baselines = [estimate_baseline(r.image) for r in refs]
        self.report = WriteReport()
        self._calls = 0
        if debug_dir:
            os.makedirs(os.path.join(debug_dir, "pieces"), exist_ok=True)
            for i, r in enumerate(refs):
                r.image.save(os.path.join(debug_dir, f"style_{i}.png"))
                with open(os.path.join(debug_dir, f"style_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(r.text)

    # ------------------------------------------------------------------ public ---------
    def write(self, lines: List[str]) -> Dict[int, WrittenLine]:
        """lines: text per line ('' = blank). Returns {line index: WrittenLine} for non-empty lines."""
        pieces: Dict[int, List[Piece]] = {}
        k = 0
        for i, text in enumerate(lines):
            if text.strip():
                ref = pick_ref(text, self.refs, k)
                pieces[i] = [Piece(i, t, ref) for t in split_pieces(text, self.wcfg.max_piece_chars)]
                k += 1
        all_pieces = [p for ps in pieces.values() for p in ps]
        self.report.pieces = len(all_pieces)
        self._total = max(1, len(all_pieces) * self.wcfg.candidates)

        self._run(all_pieces, self.wcfg.candidates, "Writing")
        for rnd in range(self.wcfg.retries):
            failed = [p for p in all_pieces if not p.ok]
            if not failed:
                break
            if rnd == self.wcfg.retries - 1:  # last round: split stubborn multi-word pieces
                failed = self._split_failed(pieces, failed)
                all_pieces = [p for ps in pieces.values() for p in ps]
            self.report.regenerated += len(failed)
            self._total += 2 * len(failed)
            self._run(failed, 2, f"Re-writing {len(failed)} piece(s) that didn't pass the check")

        out = {i: self._finish_line(i, ps) for i, ps in pieces.items()}
        self.report.pieces = len(all_pieces)
        self.report.model_calls = self._calls
        self.report.model_error = getattr(self.gen, "last_error", None)
        if self.debug_dir:
            self._save_report(all_pieces)
        return out

    # ------------------------------------------------------------------ internals ------
    def _run(self, todo: List[Piece], n: int, msg: str):
        """Generate n attempts for each piece (batched per style ref) and keep the best."""
        jobs = [p for p in todo for _ in range(n)]
        by_ref: Dict[int, List[Piece]] = {}
        for p in jobs:
            by_ref.setdefault(p.ref, []).append(p)
        for r, items in by_ref.items():
            for b in range(0, len(items), self.wcfg.batch_size):
                chunk = items[b: b + self.wcfg.batch_size]
                imgs = self.gen.generate_many([p.text for p in chunk], self.refs[r], self.seed + 7919 * self._calls)
                self._calls += 1
                reads = self._read(imgs, chunk)
                for p, img, read in zip(chunk, imgs, reads):
                    self._consider(p, img, read)
                self.progress(min(0.99, self._done_frac(len(chunk))), msg)

    def _done_frac(self, k: int) -> float:
        self._done = getattr(self, "_done", 0) + k
        return self._done / self._total

    def _read(self, imgs, pieces) -> List[Optional[str]]:
        if not self.ocr:
            return [None] * len(imgs)
        todo = [i for i, p in enumerate(pieces) if len(re.sub(r"[^A-Za-z0-9]", "", p.text)) >= 3]
        out: List[Optional[str]] = [None] * len(imgs)
        if todo:
            res = self.ocr.read([trim_width(imgs[i]) for i in todo])
            for i, r in zip(todo, res):
                out[i] = r.text
        return out

    def _consider(self, p: Piece, img: Image.Image, read: Optional[str]):
        p.attempts += 1
        ratio = 1.0
        cer = None
        if self.verify:
            expected = len(p.text) * self.ppc[p.ref]
            ratio = trim_width(img, pad=0).width / max(1.0, expected)
            if img.width <= 16 and not ink_mask(np.array(img)).any():
                ratio = 0.0
            if read is not None:
                cer = loose_cer(read, p.text)
        lo, hi = self.wcfg.min_width_ratio, self.wcfg.max_width_ratio
        width_bad = 0.0 if lo <= ratio <= hi else (lo - ratio if ratio < lo else ratio - hi)
        # run-on: the model kept writing after the text (reads as extra letters)
        # missing: letters or whole words of the text that never got written (a dropped word
        # is only ~0.25 CER on a 4-word piece, so the CER limit alone lets it through)
        extra = missing = 0.0
        miss_n = None
        if read is not None:
            n = len(letters(p.text))
            extra = max(0, len(letters(read)) - n - max(2, int(0.15 * n))) / max(1, n)
            allowed = int(self.wcfg.max_missing * n)
            miss_n = missing_letters(read, p.text)
            missing = max(0, miss_n - allowed) / max(1, n)
        ok = (width_bad == 0 and extra == 0 and missing == 0
              and (cer is None or cer <= self.wcfg.max_cer))
        score = (cer or 0.0) + 2 * width_bad + extra + 2 * missing
        if self.debug_dir:
            tag = f"l{p.line:03d}_{re.sub(r'[^A-Za-z0-9]+', '-', p.text)[:24]}_try{p.attempts}_{'ok' if ok else 'bad'}"
            img.save(os.path.join(self.debug_dir, "pieces", tag + ".png"))
        if score < p.score:
            p.image, p.score, p.ok, p.ocr, p.cer, p.width_ratio = img, score, ok, read or "", cer, ratio
            p.missing = miss_n

    def _split_failed(self, pieces: Dict[int, List[Piece]], failed: List[Piece]) -> List[Piece]:
        again = []
        for p in failed:
            words = p.text.split()
            if len(words) < 2:
                again.append(p)
                continue
            self.report.split += 1
            h = len(words) // 2
            halves = [Piece(p.line, " ".join(words[:h]), p.ref), Piece(p.line, " ".join(words[h:]), p.ref)]
            lst = pieces[p.line]
            j = lst.index(p)
            pieces[p.line] = lst[:j] + halves + lst[j + 1:]
            again.extend(halves)
        return again

    def _finish_line(self, i: int, ps: List[Piece]) -> WrittenLine:
        ref = ps[0].ref
        imgs = []
        for p in ps:
            if not p.ok:
                self.report.unverified += 1
            if p.image is None or p.width_ratio < 0.3:  # the model produced (almost) nothing
                p.image, p.fallback = self._font_piece(p), True
                self.report.font_fallback += 1
            imgs.append(p.image)
        line = stitch(imgs, self.gaps[ref]) if len(imgs) > 1 else trim_width(imgs[0], pad=4)
        text = " ".join(p.text for p in ps)
        # the model continues the style image, so it keeps that image's baseline; measure it on
        # the line itself when there is enough text, else trust the style line
        baseline = estimate_baseline(line) if len(text) >= 12 else self.baselines[ref]
        if abs(baseline - self.baselines[ref]) > STYLE_HEIGHT * 0.2:
            baseline = self.baselines[ref]
        if self.debug_dir:
            line.save(os.path.join(self.debug_dir, f"line_{i:03d}.png"))
        return WrittenLine(line, baseline)

    def _font_piece(self, p: Piece) -> Image.Image:
        """Last resort so no words go missing; scaled to the handwriting's width."""
        if not hasattr(self, "_font"):
            self._font = FontGenerator(jitter=0.5)
        img = trim_width(self._font.generate(p.text, self.refs[p.ref], self.seed))
        target = max(8, int(len(p.text) * self.ppc[p.ref]))
        return img.resize((target, img.height), Image.LANCZOS)

    def _save_report(self, pieces: List[Piece]):
        path = os.path.join(self.debug_dir, "pieces.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["line", "text", "style", "attempts", "passed", "ocr_read", "loose_cer", "width_ratio",
                        "missing_letters", "font_fallback"])
            for p in pieces:
                w.writerow([p.line, p.text, p.ref, p.attempts, p.ok, p.ocr,
                            "" if p.cer is None else f"{p.cer:.2f}", f"{p.width_ratio:.2f}", "" if p.missing is None else p.missing, p.fallback])
