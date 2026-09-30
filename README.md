# ✍️ CopyWrite AI

Give it your notes as text and a photo of a few lines of your handwriting. It writes the notes onto ruled, plain, grid or photographed pages **in your handwriting**, then exports PNG + PDF.

- **AI model:** [Emuru](https://huggingface.co/blowing-up-groundhogs/emuru) (CVPR 2025), a VAE + T5 Transformer that imitates an unseen handwriting from a single line (zero-shot).
- **Handwriting OCR:** [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) reads your sample automatically; you only fix its mistakes.
- **Classical layout engine:** word wrap, ruled-line detection, baseline alignment, natural jitter, ink blending.
- **Baseline for comparison:** a jittered handwriting font (no AI).
- **Optional** invisible watermark.

Full design: [`docs/DESIGN.md`](docs/DESIGN.md)

## Quick start

### A) On a free GPU (recommended, real results)
1. Push this repo to GitHub.
2. Open [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb) in Google Colab (or Kaggle) with a **T4 GPU**.
3. Set `REPO_URL`, run all cells, and open the `*.gradio.live` link it prints.

### B) On your laptop (no GPU, font baseline, good for development)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python app.py --backend font      # open http://127.0.0.1:7860
```
To try the AI model on CPU (slow, needs ~8 GB free RAM): `pip install -r requirements-model.txt` and pick `emuru` in the UI.

### Command line
```bash
python -m copywrite.cli --text samples/text/sample_notes.txt \
  --style samples/style/me.jpg \
  --backend emuru --background ruled --ink blue --out outputs/run1
```
Without `--style-text`, OCR reads your sample and prints what it read; save a corrected copy and pass it with `--style-text me.txt` if needed. Add `--watermark` for the optional watermark (needs `pip install invisible-watermark`).

## How to take a good style photo
- 2-5 lines of your normal writing, 4-8 words each, in the pen you want to imitate.
- Shoot from directly above in even light.
- Click **"Read my handwriting"**: the lines are detected and read by OCR. Fix any wrong words in the text box (flagged lines first). The text must match your writing exactly.

## Evaluation
```bash
pip install -r requirements-model.txt -r requirements-extra.txt
python scripts/evaluate.py --photos me_12lines.jpg --text me_12lines.txt --k 2 --backends emuru font --out outputs/eval_me
```
This writes per-line OCR CER (TrOCR), KID/FID vs. your real lines, seconds per line, and `comparison_sheet.png` (real vs. AI vs. font).

## Project structure
```
app.py                      Gradio web UI
copywrite/
  preprocess.py             photo -> clean 64px style lines
  ocr.py                    TrOCR auto-transcription of your sample (+ confidence)
  backends.py               EmuruGenerator (AI) / FontGenerator (baseline)
  background.py             page templates, ruled-line + margin detection
  layout.py                 wrapping, placement, jitter, ink compositing
  pipeline.py               end-to-end orchestration
  watermark.py              optional invisible watermark
  cli.py                    command line
scripts/evaluate.py         metrics + comparison sheet
notebooks/run_on_colab.ipynb
docs/DESIGN.md              system design document
samples/                    example text; put your handwriting photos in samples/style (git-ignored)
```

## Credits
Emuru by Pippi, Quattrini, Cascianelli, Tonioni and Cucchiara (AImageLab, CVPR 2025, MIT license). TrOCR by Microsoft (MIT).
