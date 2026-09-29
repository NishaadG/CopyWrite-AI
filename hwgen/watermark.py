"""OPTIONAL invisible watermark (off by default; turn on with the checkbox / --watermark).

Uses the `invisible-watermark` package (DWT-DCT frequency-domain embedding, the same one
Stable Diffusion used). Survives mild JPEG compression / resizing; does NOT survive
print-and-rescan - worth a line in the reflective journal.

    pip install invisible-watermark
"""

import numpy as np
from PIL import Image

PAYLOAD_BYTES = 8


def _payload(text: str) -> bytes:
    return text.encode("utf-8")[:PAYLOAD_BYTES].ljust(PAYLOAD_BYTES, b"\0")


def embed(page: Image.Image, text: str = "HWGEN") -> Image.Image:
    try:
        from imwatermark import WatermarkEncoder
    except ImportError as e:
        raise ImportError("Watermark is on but 'invisible-watermark' isn't installed: "
                          "pip install invisible-watermark") from e
    bgr = np.array(page.convert("RGB"))[:, :, ::-1].copy()
    enc = WatermarkEncoder()
    enc.set_watermark("bytes", _payload(text))
    out = enc.encode(bgr, "dwtDct")
    return Image.fromarray(out[:, :, ::-1])


def detect(page: Image.Image) -> str:
    from imwatermark import WatermarkDecoder

    bgr = np.array(page.convert("RGB"))[:, :, ::-1].copy()
    dec = WatermarkDecoder("bytes", PAYLOAD_BYTES * 8)
    raw = dec.decode(bgr, "dwtDct")
    return raw.rstrip(b"\0").decode("utf-8", errors="replace")
