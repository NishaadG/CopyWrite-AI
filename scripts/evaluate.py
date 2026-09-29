"""Evaluation for the report ("Implementation & Results").

Protocol (per writer):
  1. The writer writes ~10-15 lines by hand. Photograph them, type the transcription.
  2. The first K lines are the STYLE references given to the model.
  3. The remaining lines are the REAL test set. We ask each backend to write the *same*
     sentences, so real and generated lines can be compared directly.

Metrics:
  - CER (character error rate) of a handwriting OCR model (TrOCR) on each line -> legibility.
    The CER on the REAL lines is the reference point.
  - KID / FID (torchmetrics) between generated and real line images -> realism.
    With few lines, report KID (FID is unreliable below a few hundred images).
  - seconds per line -> speed / feasibility.
  - a side-by-side sheet (real vs AI vs font) to use in slides and for the human study.

Usage:
  python scripts/evaluate.py --photos writer1.jpg --text writer1.txt --k 2 \
      --backends emuru font --out outputs/eval_writer1
Extra deps (Colab already has most): pip install torchmetrics[image] torch-fidelity
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hwgen.backends import get_generator  # noqa: E402
from hwgen.io_utils import read_text_file  # noqa: E402
from hwgen.layout import trim_width  # noqa: E402
from hwgen.preprocess import prepare_style_refs  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    return levenshtein(pred.strip(), truth.strip()) / max(1, len(truth.strip()))


class OCR:
    def __init__(self, model_id="microsoft/trocr-base-handwritten"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, img: Image.Image) -> str:
        px = self.proc(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(px, max_new_tokens=96)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0]


def to_fixed(img: Image.Image, w=512, h=64) -> np.ndarray:
    img = trim_width(img.convert("L"))
    canvas = Image.new("L", (w, h), 255)
    scale = min(1.0, w / img.width)
    img = img.resize((max(1, int(img.width * scale)), h))
    canvas.paste(img, (0, 0))
    return np.array(canvas.convert("RGB"))


def kid_fid(real, fake):
    try:
        import torch
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return {"kid": None, "fid": None, "note": "pip install torchmetrics[image] torch-fidelity"}
    r = torch.tensor(np.stack([to_fixed(i) for i in real])).permute(0, 3, 1, 2)
    f = torch.tensor(np.stack([to_fixed(i) for i in fake])).permute(0, 3, 1, 2)
    n = min(len(real), len(fake))
    out = {}
    if n < 2:
        return {"note": "need >= 2 test lines for KID"}
    kid = KernelInceptionDistance(subset_size=max(2, min(50, n)))
    kid.update(r, real=True)
    kid.update(f, real=False)
    m, s = kid.compute()
    out["kid_mean"], out["kid_std"] = float(m), float(s)
    try:
        fid = FrechetInceptionDistance(feature=64)  # small feature dim: fewer samples needed
        fid.update(r, real=True)
        fid.update(f, real=False)
        out["fid64"] = float(fid.compute())
    except Exception as e:  # too few samples -> singular covariance
        out["fid64"] = None
        out["fid_note"] = str(e)[:120]
    return out


def sheet(rows, path):
    """rows: list of (label, [images]) -> one tall comparison image."""
    tiles = []
    for label, imgs in rows:
        for im in imgs:
            tiles.append((label, trim_width(im.convert("L"))))
    w = min(1400, max(t.width for _, t in tiles) + 110)
    out = Image.new("L", (w, 70 * len(tiles)), 255)
    from PIL import ImageDraw

    d = ImageDraw.Draw(out)
    for k, (label, t) in enumerate(tiles):
        d.text((4, 70 * k + 28), label, fill=0)
        out.paste(t.crop((0, 0, min(t.width, w - 110), t.height)), (105, 70 * k + 3))
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--text", required=True, help="transcription file, one line per handwritten line")
    ap.add_argument("--k", type=int, default=2, help="how many lines to use as style reference")
    ap.add_argument("--backends", nargs="+", default=["emuru", "font"])
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = prepare_style_refs(args.photos, read_text_file(args.text))
    if len(lines) <= args.k:
        raise SystemExit(f"Need more than k={args.k} lines; found {len(lines)}.")
    refs, test = lines[: args.k], lines[args.k:]
    print(f"{len(refs)} style lines, {len(test)} real test lines")

    ocr = None if args.no_ocr else OCR()
    results, summary = [], {}
    real_imgs = [t.image for t in test]
    gen_imgs = {}

    for bname in ["real"] + args.backends:
        if bname == "real":
            imgs, secs = real_imgs, 0.0
        else:
            gen = get_generator(bname)
            if hasattr(gen, "load"):
                gen.load()
            t0 = time.time()
            imgs = []
            for i, t in enumerate(test):
                imgs.append(gen.generate(t.text, refs[i % len(refs)], seed=i))
            secs = (time.time() - t0) / len(test)
            gen_imgs[bname] = imgs
        cers = []
        for i, (t, im) in enumerate(zip(test, imgs)):
            im.save(os.path.join(args.out, f"{bname}_{i:02d}.png"))
            pred = ocr(im) if ocr else ""
            c = cer(pred, t.text) if ocr else None
            cers.append(c)
            results.append({"source": bname, "idx": i, "truth": t.text, "ocr": pred, "cer": c})
        summary[bname] = {
            "mean_cer": float(np.mean(cers)) if ocr else None,
            "seconds_per_line": round(secs, 3),
        }
        if bname != "real":
            summary[bname].update(kid_fid(real_imgs, imgs))
        print(bname, summary[bname])

    with open(os.path.join(args.out, "per_line.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    n = min(4, len(test))
    rows = [("style ref", [r.image for r in refs]), ("real", real_imgs[:n])]
    rows += [(b, gen_imgs[b][:n]) for b in args.backends]
    sheet(rows, os.path.join(args.out, "comparison_sheet.png"))
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
