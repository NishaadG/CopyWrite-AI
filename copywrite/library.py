"""Ready-made handwriting styles, so you can try the app without photographing your own.

Source: the IAM handwriting database, via the Hugging Face dataset `Teklia/IAM-line`
(real English handwriting from ~650 writers, one text line per image, with an exact
transcription). We download one split (~24 MB, once), pick a few dozen clean lines from
different writers, and turn each into a StyleRef exactly like a user's own photo.

Why this works: Emuru only needs ONE line image + its text to imitate a handwriting, and
IAM already provides both, so no OCR or typing is needed.

The images are downloaded at runtime and cached in ~/.cache/copywrite/library; they are
never committed to the repo (the IAM terms allow research / non-commercial use).
"""

import io
import json
import os
import re
from typing import List

from PIL import Image

from .preprocess import MAX_STYLE_WIDTH, StyleRef, single_ref

DATASET = "Teklia/IAM-line"
SPLIT_FILE = "data/validation.parquet"
N_STYLES = 36
CACHE_VERSION = 1  # bump when the preprocessing changes, so old cached styles are rebuilt
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "copywrite", f"library_v{CACHE_VERSION}")
_ALLOWED = re.compile(r"^[A-Za-z0-9 ,.;:!?'()\-]+$")


def clean_text(text: str) -> str:
    """IAM transcriptions are tokenised ('chosen ,'); rejoin punctuation the way it was written."""
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text.strip())
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s+", " ", text)


def _usable(text: str) -> bool:
    if not (25 <= len(text) <= 60) or not _ALLOWED.match(text) or "..." in text:
        return False
    return not re.search(r"\b[A-Z] [A-Z]\b", text)  # spelled-out initials like 'B B C'


def _download() -> List[StyleRef]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    table = pq.read_table(hf_hub_download(DATASET, SPLIT_FILE, repo_type="dataset"))
    texts = [clean_text(t) for t in table.column("text").to_pylist()]
    candidates = [i for i, t in enumerate(texts) if _usable(t)]
    # consecutive rows come from the same page/writer, so visit rows spread across the split
    step = max(1, len(candidates) // N_STYLES)
    order = [i for off in range(step) for i in candidates[off::step]]
    images = table.column("image").to_pylist()
    refs = []
    for i in order:
        ref = single_ref(Image.open(io.BytesIO(images[i]["bytes"])), texts[i], remove_lines=False)
        if ref.image.width <= MAX_STYLE_WIDTH:  # longer refs make generation slow
            refs.append(ref)
            if len(refs) == N_STYLES:
                break
    return refs


def load_library() -> List[StyleRef]:
    """Return the library styles, downloading and caching them on first use."""
    index = os.path.join(CACHE_DIR, "index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            entries = json.load(f)
        return [StyleRef(Image.open(os.path.join(CACHE_DIR, e["file"])).convert("L"), e["text"])
                for e in entries]
    refs = _download()
    os.makedirs(CACHE_DIR, exist_ok=True)
    entries = []
    for k, ref in enumerate(refs):
        name = f"style_{k:02d}.png"
        ref.image.save(os.path.join(CACHE_DIR, name))
        entries.append({"file": name, "text": ref.text})
    with open(index, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=1)
    return refs
