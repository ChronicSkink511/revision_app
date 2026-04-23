# Local Engineering Revision Assistant

A fully local, resource-conscious Python application for revision generation from engineering documents and images.

## Features

- Multi-file upload for `PDF`, `DOCX`, `PPTX`, `TXT`, `MD`, and common image formats.
- ZIP ingestion for folder-style uploads with recursive extraction of supported files.
- Safe parsing of untrusted uploads with corruption handling and warnings.
- Engineering image analysis using:
  - OCR (`pytesseract`) for formulas, labels, and text.
  - Lightweight CV features (`opencv`) for lines/contours/circles.
  - Local LLM interpretation of extracted image signals.
- Topic detection from headings and corpus content.
- Concise revision notes per topic.
- Quiz generation per topic (MCQ + short-answer).
- Export generated notes/quizzes to `PDF`, `DOCX`, `JSON`, `CSV`.
- Fully local inference via `llama-cpp-python` and a local GGUF model.
- Windows-friendly fallback local backend via `gpt4all` for GGUF loading when `llama-cpp-python` is unavailable.

## Project Structure

revision_local_app/
- bootstrap.py
- streamlit_app.py
- requirements.txt
- README.md
- scripts/
  - run_app.py
- data/
  - models/
  - uploads/
  - work/
- logs/
- src/revision_app/
  - config.py
  - logging_utils.py
  - schemas.py
  - ingestion/
  - parsing/
  - image_analysis/
  - llm/
  - analysis/
  - export/

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional local-LLM extras (for full offline model inference on local machines):

```powershell
pip install -r requirements-local-llm.txt
```

3. Ensure Tesseract OCR is installed on your machine.

Windows example:
- Install Tesseract from a trusted source.
- Optionally set:

```powershell
$env:TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

4. Download a small GGUF model (1-4 GB) into `data/models`.

Recommended examples:
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- Any compact instruct model supported by `llama.cpp`.

On Windows, if `llama-cpp-python` cannot build due to missing C++ toolchain, the app will use `gpt4all` automatically with the same local GGUF file.

You can set the model filename with:

```powershell
$env:REVAPP_GGUF_MODEL = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
```

## Run

Quick setup helper:

```powershell
python scripts/setup_env.py
```

Then run:

```powershell
python bootstrap.py
python scripts/run_app.py
```

Or directly:

```powershell
streamlit run streamlit_app.py
```

## Streamlit Community Cloud Deploy

- Push the repository to GitHub.
- In Streamlit Community Cloud, set app entrypoint to `streamlit_app.py` at repository root.
- A root `requirements.txt` is included and points to `revision_local_app/requirements.txt`.
- Cloud deployment uses lightweight defaults and fallback generation when local GGUF backends are unavailable.
- For local desktop use, install `requirements-local-llm.txt` to enable full local model backends.

## Settings Menu

- Open the app sidebar and expand `Settings`.
- Change model/runtime/options and click `Save Settings`.
- Use `Reset to Defaults` to restore default values.
- Settings are persisted to `data/work/app_settings.json`.
- Default resource mode is controlled from this menu and starts as `Low`.

## Configurable Environment Variables

- `REVAPP_GGUF_MODEL` (default: `Qwen2.5-3B-Instruct-Q4_K_M.gguf`)
- `REVAPP_CTX` (default: `1024`)
- `REVAPP_MAX_TOKENS` (default: `256`)
- `REVAPP_THREADS` (default: `4`)
- `REVAPP_TEMP` (default: `0.2`)
- `REVAPP_ENABLE_EMBEDDINGS` (default: `false`)
- `REVAPP_EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `REVAPP_MAX_TOPICS` (default: `6`)
- `REVAPP_QUIZ_PER_TOPIC` (default: `6`)
- `TESSERACT_CMD` (optional absolute path)

## Low-Resource Notes

- Keep GGUF model size around 1-4 GB.
- Keep context size low (default 1024).
- Embeddings are optional and disabled by default.
- Streamlit UI resource mode defaults to `Low` for minimal CPU/RAM usage.
- Parsing and analysis use truncation and chunking to limit memory usage.
- If model loading fails, fallback heuristics still generate useful notes and quizzes.

## Security

- Uploaded files are treated as untrusted input.
- No file contents are executed.
- ZIP extraction is path-sanitized and size-limited.
- Corrupted or unsupported files are skipped with clear warnings.

## Disclaimer

This app is for study support and revision assistance. Always validate critical engineering answers against trusted references.
