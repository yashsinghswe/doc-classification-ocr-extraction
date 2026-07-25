# Document Classification & OCR / Information Extraction

A 6-phase document processing pipeline for healthcare documents — OCR, classification, extraction, validation, and human-in-the-loop review queue.

## Overview

This pipeline takes scanned images or digital PDFs of healthcare documents and automatically:
1. Extracts text via OCR or PDF parsing
2. Classifies the document type using semantic embeddings
3. Extracts structured fields using rules or an LLM
4. Validates the extracted data for logical consistency
5. Auto-approves high-confidence results or routes to a human review queue

---

## Supported Document Types

| Type | Extraction Method |
|---|---|
| `patient_registration` | Rule-based regex |
| `medical_invoice` | Rule-based regex |
| `appointment_letter` | Gemini LLM |
| `discharge_summary` | Gemini LLM |
| `lab_report` | Gemini LLM |

---

## Architecture

```
Input (PDF / Image)
       │
       ▼
Phase 1 — OCR (EasyOCR)              ← scanned images / image-based PDFs
Phase 2 — PDF Extraction              ← digital PDFs (pdfplumber / pymupdf)
       │
       ▼
Phase 3 — Classification              ← few-shot embedding search (all-MiniLM-L6-v2)
       │
       ▼
Phase 4 — Information Extraction      ← regex rules + Gemini LLM (tiered)
       │
       ▼
Phase 5 — Validation                  ← Pydantic + regex + cross-field checks
       │
       ▼
Phase 6 — Review Queue                ← auto-approve or route to human review (Flask UI)
```

---

## Project Structure

```
doc_pipeline/
├── main.py                  # Orchestrates all 6 phases
├── phase1_ocr.py            # EasyOCR for scanned images
├── phase2_pdf.py            # pdfplumber + pymupdf for digital PDFs
├── phase3_classify.py       # Few-shot embedding classifier
├── phase4_extract.py        # Tiered extraction: rules + Gemini LLM
├── phase5_validate.py       # Pydantic + regex + cross-field validation
├── phase6_queue.py          # SQLite review queue + routing logic
├── review_ui/
│   ├── app.py               # Flask review UI backend
│   └── templates/
│       └── review.html      # Review UI frontend
├── data/
│   ├── few_shot/            # Seed examples for Phase 3 classifier
│   │   ├── patient_registration/
│   │   ├── medical_invoice/
│   │   ├── appointment_letter/
│   │   ├── discharge_summary/
│   │   └── lab_report/
│   ├── scanned/             # JPG inputs for Phase 1
│   └── digital_pdf/         # PDF inputs for Phase 2
└── models/
    ├── embedding_index.json  # Phase 3 centroid index (auto-generated)
    └── review_queue.db       # SQLite queue (auto-generated)
```

---

## Setup

### 1. Create and activate virtual environment
```bash
python3 -m venv doc_pipeline_env
source doc_pipeline_env/bin/activate
```

### 2. Install dependencies
```bash
pip install easyocr pdf2image pymupdf pdfplumber \
            sentence-transformers scikit-learn numpy \
            pydantic sqlalchemy flask google-generativeai \
            Pillow reportlab
```

### 3. Install poppler (required for pdf2image)
```bash
brew install poppler   # macOS
```

### 4. Set your Gemini API key
```bash
export GEMINI_API_KEY=your-key-here
```

### 5. Seed the Phase 3 classifier
Place 3+ example `.txt` files per document type under `data/few_shot/<doc_type>/`, then run:
```bash
python phase3_classify.py --seed
```

---

## Running the Pipeline

### Process scanned images (Phase 1)
```bash
python -c "from main import process_batch; process_batch('data/scanned')"
```

### Process digital PDFs (Phase 2)
```bash
python -c "from main import process_batch; process_batch('data/digital_pdf')"
```

### Process a single file
```bash
python main.py data/scanned/LAB_REPORT_Example_1.jpg
python main.py data/digital_pdf/MEDICAL_INVOICE_Example_1.pdf
```

### Run the built-in demo (no files needed)
```bash
python main.py --demo
```

---

## Review UI

Launch the Flask review UI to handle documents that were queued for human review:
```bash
python review_ui/app.py
```

Open `http://localhost:5050` in your browser.

Documents are queued when:
- Classification confidence < 0.75
- Validation fails (e.g. admission date after discharge date)

Confirming a document in the UI automatically updates the Phase 3 classifier index via the feedback loop.

---

## Phase Details

| Phase | Tool | Purpose |
|---|---|---|
| 1 | EasyOCR | Convert scanned images to text + bounding boxes |
| 2 | pdfplumber / pymupdf | Extract text and tables from digital PDFs |
| 3 | sentence-transformers (all-MiniLM-L6-v2) | Classify document type via cosine similarity |
| 4 | Regex rules + Gemini 1.5 Flash | Extract structured fields |
| 5 | Pydantic + re + datetime | Validate extracted data |
| 6 | SQLite + Flask | Route to review queue or auto-approve |

---

## Notes

- LayoutLMv3 is stubbed in Phase 4 and not active — requires 500+ labelled documents per type to train
- The Phase 3 classifier improves automatically as reviewers confirm documents in the UI
- `models/` directory is auto-created on first run
- Data files (`*.db`, `*.json`) are excluded from version control via `.gitignore`
