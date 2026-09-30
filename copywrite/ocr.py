"""Handwriting OCR (TrOCR) - reads the user's style sample so they don't have to type it.

Why OCR + human correction instead of OCR alone:
Emuru is conditioned on the style image AND its exact transcription. If the transcription
is wrong, the model links the wrong letters to the wrong shapes and the output style
degrades. So OCR pre-fills the text, flags lines it is unsure about, and the user fixes
the few mistakes. Typically that is a couple of words, not the whole sample.

The same model checks every piece the generator writes (writer.py: does it say what it
should?) and is reused in evaluation to measure legibility (character error rate).

Model: microsoft/trocr-base-handwritten - a Vision Transformer encoder + text Transformer
decoder (encoder-decoder, like the syllabus' Module 3), fine-tuned on IAM handwriting.
~330 M params: fine on Colab, ~1-2 s/line on a laptop CPU.
"""

import threading
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
LOW_CONFIDENCE = 0.80


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0-1, geometric mean of token probabilities

    @property
    def needs_check(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


class HandwritingOCR:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise ImportError("OCR needs the model packages: pip install -r requirements-model.txt") from e
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device).eval()

    def read(self, images: List[Image.Image]) -> List[OCRResult]:
        if not images:
            return []
        px = self.proc(images=[i.convert("RGB") for i in images], return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(px, max_new_tokens=96, num_beams=4,
                                      return_dict_in_generate=True, output_scores=True)
        texts = self.proc.batch_decode(out.sequences, skip_special_tokens=True)
        confs = self._confidences(out, len(texts))
        return [OCRResult(t.strip(), c) for t, c in zip(texts, confs)]

    def _confidences(self, out, n: int) -> List[float]:
        seq_scores = getattr(out, "sequences_scores", None)
        if seq_scores is not None:  # beam search: length-normalised log-prob per sequence
            return [float(self.torch.exp(s)) for s in seq_scores]
        try:
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, out.get("beam_indices"), normalize_logits=False)
            confs = []
            for row in scores:
                row = row[self.torch.isfinite(row)]
                confs.append(float(self.torch.exp(row.mean())) if len(row) else 0.0)
            return confs
        except Exception:
            return [1.0] * n  # confidence unavailable in this transformers version

    def __call__(self, img: Image.Image) -> str:
        return self.read([img])[0].text


_OCR = {}
_OCR_LOCK = threading.Lock()


def get_ocr(model_id: str = DEFAULT_MODEL) -> HandwritingOCR:
    with _OCR_LOCK:  # the app preloads it in a background thread
        if model_id not in _OCR:  # load once, reuse
            _OCR[model_id] = HandwritingOCR(model_id)
    return _OCR[model_id]


def transcribe_lines(line_images: List[Image.Image], model_id: str = DEFAULT_MODEL) -> List[OCRResult]:
    return get_ocr(model_id).read(line_images)
