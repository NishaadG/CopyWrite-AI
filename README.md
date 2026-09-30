# CopyWrite AI

Give it your notes as text and a photo of a few lines of your handwriting (or pick a sample handwriting). It writes the notes onto ruled, plain, grid or photographed pages **in that handwriting**, then exports PNG + PDF.

- **AI model:** [Emuru](https://huggingface.co/blowing-up-groundhogs/emuru) (CVPR 2025), a VAE + T5 Transformer that imitates an unseen handwriting from a single line (zero-shot).
- **Reliable generation:** each line is written in short pieces, every piece is read back by OCR and rewritten if words are missing or wrong, then the pieces are stitched together. No text goes missing.
- **Sample handwritings:** 36 real writers from the [IAM database](https://huggingface.co/datasets/Teklia/IAM-line), so you can try it without photographing your own.
- **Handwriting OCR:** [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) reads your sample automatically (you only fix its mistakes) and checks the generated text.
- **Classical layout engine:** word wrap, ruled-line detection, baseline alignment, natural jitter, ink blending.
- **Baseline for comparison:** a jittered handwriting font (no AI).
- **Optional** invisible watermark.

Full design: [`docs/DESIGN.md`](docs/DESIGN.md)

## How to run it

### Option A: Google Colab (free GPU, real AI handwriting). Use this for results and the demo.

1. Click this link: **[Open the notebook in Colab](https://colab.research.google.com/github/NishaadG/CopyWrite-AI/blob/main/notebooks/run_on_colab.ipynb)**. Sign in with a Google account if asked.
2. In Colab: **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all** (`Ctrl+F9`). If Colab warns "This notebook was not authored by Google", click **Run anyway**.
4. Wait 3-5 minutes. The last cell prints `Running on public URL: https://xxxx.gradio.live`. **Click that link.**
5. Use the app (below). Leave the Colab tab open; closing it stops the app after a while.

**After you change the code:** `git push` from your laptop, then in Colab do **Runtime → Restart session and run all**. The notebook pulls the latest code automatically.

### Option B: your laptop (no GPU needed; shows a simple font instead of the AI model)
Good for trying the interface and working on the layout. The AI model is switched off automatically when there is no GPU.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                      # then open http://127.0.0.1:7860 in your browser
```

## Using the app

1. **Choose a handwriting**
   - **Sample handwritings** tab: 36 real handwritings from the IAM database. Click one; the highlighted one is used. No photo or typing needed.
   - **My own handwriting** tab: upload a photo of 2-5 lines of your writing, click **Read my handwriting**, and fix any wrong words in the text box. The text must match your writing exactly.
2. **What should it write?** Paste your notes (an example is already filled in) or upload a `.txt` / `.docx`. Each new line starts a new paragraph.
3. **Page:** choose ruled / plain / grid paper (or a photo of a real page), the ink colour, the handwriting size and the **Quality**:
   - **Fast:** one attempt per piece, and only pieces that fail the check are rewritten.
   - **Best:** three attempts per piece, and the one the OCR reads best is kept. Slower, but cleaner.

   **More options** has page size, line spacing, margin, messiness, engine, seed and **Save debug images**.
4. Click **Write my notes**. The pages appear on the right; **Download PDF** is below them. The message above the pages says how many pieces were rewritten and whether any never passed the check.

Tips:
- If the writing is too big or small for the lines, change **Handwriting size**.
- Different samples write at different widths and some are easier for the model, so try a few.
- The first generation on Colab is slow (the models are loading), and later ones are faster. A page takes roughly 1-3 minutes on a T4.

**If the output looks wrong:** tick **More options → Save debug images**, generate again, and download the zip. It contains:
- the style line used
- every piece and every attempt, marked `ok` or `bad`
- each stitched line and the final pages
- `pieces.csv`: what the OCR read, the error rate and the width check for every piece

Send the zip along with the **Run details** text.

### Command line
```bash
python -m copywrite.cli --text samples/text/sample_notes.txt \
  --style samples/style/me.jpg \
  --backend emuru --background ruled --ink blue --quality best --debug-dir outputs/run1/debug --out outputs/run1
```
Without `--style-text`, OCR reads your sample and prints what it read. Save a corrected copy and pass it with `--style-text me.txt` if needed. Add `--watermark` for the optional watermark (needs `pip install invisible-watermark`).

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
  ocr.py                    TrOCR: reads your sample (+ confidence) and checks generated text
  library.py                sample handwritings (IAM lines, downloaded + cached on first use)
  backends.py               EmuruGenerator (AI) / FontGenerator (baseline): one model call
  writer.py                 LineWriter: pieces -> batched generation -> OCR check -> retries -> stitch
  metrics.py                CER / letter-level CER
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
Emuru by Pippi, Quattrini, Cascianelli, Tonioni and Cucchiara (AImageLab, CVPR 2025, MIT license). TrOCR by Microsoft (MIT). Sample handwritings from the IAM Handwriting Database (Marti & Bunke, 2002) via `Teklia/IAM-line`; downloaded at runtime, not redistributed here.
