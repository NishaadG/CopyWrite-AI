"""Reading input documents and writing output pages."""

import os
from typing import List

from PIL import Image


def read_text_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        import docx  # python-docx

        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def save_pages(pages: List[Image.Image], out_dir: str, dpi: int = 150, stem: str = "notes") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    pngs = []
    for i, p in enumerate(pages, 1):
        path = os.path.join(out_dir, f"{stem}_page{i:02d}.png")
        p.save(path, dpi=(dpi, dpi))
        pngs.append(path)
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    rgb = [p.convert("RGB") for p in pages]
    rgb[0].save(pdf, save_all=True, append_images=rgb[1:], resolution=float(dpi))
    return {"pngs": pngs, "pdf": pdf}
