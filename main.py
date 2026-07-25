"""
doc_pipeline/main.py
────────────────────
Orchestrates all six phases for a single document file.

Usage
-----
    python main.py <path_to_document>
    python main.py invoice.pdf
    python main.py scan.jpg
    python main.py --demo          # runs against a built-in text sample

Environment
-----------
    GEMINI_API_KEY      – required for LLM extraction in Phase 4
                          (rule-based fallback used if not set)
"""

import json
import os
import sys
from pathlib import Path

from phase2_pdf import is_digital_pdf
from phase3_classify import FewShotClassifier
from phase4_extract import extract
from phase5_validate import validate
from phase6_queue import should_route_to_review, add_to_queue, get_stats

# Shared classifier instance (loads/saves from models/embedding_index.json)
_classifier = FewShotClassifier()


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_document(
    file_path: str,
    use_llm: bool = True,
    api_key: str = None,
) -> dict:
    """
    Run the full 6-phase pipeline on a single document.

    Parameters
    ----------
    file_path : path to a PDF or image file
    use_llm   : whether to call Claude for Phase 4 extraction
    api_key   : Anthropic API key (falls back to env var ANTHROPIC_API_KEY)

    Returns
    -------
    {
        "file":       str,
        "outcome":    "auto_approved" | "queued_for_review",
        "review_id":  str | None,
        "doc_type":   str,
        "confidence": float,
        "extracted":  dict,
        "validation": {passed, errors, warnings},
        "text_length": int,
    }
    """
    _banner(f"Processing: {file_path}")

    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    ext = path.suffix.lower()
    words_with_bbox = []

    # ── Phase 1 / Phase 2 — Text extraction ──────────────────────────────────
    if ext == ".pdf":
        if is_digital_pdf(file_path):
            _log("Phase 2", "Digital PDF → pdfplumber / pymupdf")
            from phase2_pdf import extract_digital_pdf
            raw = extract_digital_pdf(file_path)
        else:
            _log("Phase 1", "Scanned PDF → EasyOCR")
            from phase1_ocr import run_easyocr
            raw = run_easyocr(pdf_path=file_path)

    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        _log("Phase 1", "Image file → EasyOCR")
        from phase1_ocr import run_easyocr
        raw = run_easyocr(image_path=file_path)

    else:
        return {"error": f"Unsupported file type: {ext}"}

    full_text       = raw.get("full_text", "")
    words_with_bbox = raw.get("words", [])
    _log("Extracted", f"{len(full_text):,} chars · {len(words_with_bbox):,} words")

    if not full_text.strip():
        return {"error": "No text extracted from document"}

    # ── Phase 3 — Classification ──────────────────────────────────────────────
    _log("Phase 3", "Classifying …")
    classification = _classifier.classify(full_text)
    doc_type   = classification["predicted_type"]
    confidence = classification["confidence"]
    _log("Phase 3", f"→ '{doc_type}'  confidence={confidence:.3f}")

    # ── Phase 4 — Extraction ──────────────────────────────────────────────────
    _log("Phase 4", f"Extracting fields for '{doc_type}' …")
    if doc_type in ("uncertain", "unknown"):
        extracted = {
            "warning":  "Classification uncertain — extraction skipped",
            "_doc_type": doc_type,
        }
    else:
        extracted = extract(
            text=full_text,
            doc_type=doc_type,
            words_with_bbox=words_with_bbox,
            use_llm=use_llm,
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        )
    public_fields = [k for k in extracted if not k.startswith("_")]
    _log("Phase 4", f"→ {len(public_fields)} field(s) extracted")

    # ── Phase 5 — Validation ──────────────────────────────────────────────────
    _log("Phase 5", "Validating …")
    validation = validate(extracted, doc_type)

    # ── Phase 6 — Route decision ──────────────────────────────────────────────
    needs_review, reason = should_route_to_review(classification, validation)

    if needs_review:
        _log("Phase 6", f"QUEUED FOR REVIEW — {reason}")
        review_id = add_to_queue(
            file_path=file_path,
            full_text=full_text,
            classification=classification,
            extraction=extracted,
            validation=validation,
            route_reason=reason,
        )
        outcome = "queued_for_review"
    else:
        _log("Phase 6", "AUTO-APPROVED → structured JSON output")
        review_id = None
        outcome   = "auto_approved"

    return {
        "file":        path.name,
        "outcome":     outcome,
        "review_id":   review_id,
        "doc_type":    doc_type,
        "confidence":  round(confidence, 4),
        "extracted":   extracted,
        "validation": {
            "passed":   validation["passed"],
            "errors":   validation["errors"],
            "warnings": validation["warnings"],
        },
        "text_length": len(full_text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing helper
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(folder: str, use_llm: bool = True) -> list[dict]:
    """Process all PDFs and images in a folder."""
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    files = [
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in supported
    ]
    print(f"\nBatch: {len(files)} file(s) in '{folder}'")
    results = []
    for f in files:
        result = process_document(str(f), use_llm=use_llm)
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demo (no file needed)
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_INVOICE = """
ACME Supplies Pvt Ltd
123, Industrial Area, Andheri East, Mumbai - 400069
GSTIN: 27AAAPL1234C1Z5  |  PAN: AAAPL1234C

INVOICE
Invoice No : INV-2024-0042
Invoice Date: 15/07/2024
Due Date   : 30/07/2024

Bill To:
Zeta Enterprises Ltd
456 Commerce Street, Pune - 411001

Description          Qty    Unit Price    Total
-------------------------------------------------
Widget A              10      500.00      5,000.00
Widget B               5    1,000.00      5,000.00
-------------------------------------------------
Subtotal                                10,000.00
GST @ 18%                                1,800.00
Grand Total                             11,800.00  INR

Payment Terms: Net 15 days.  Bank: HDFC, A/C 0012345678
"""


def run_demo():
    """Run the pipeline on a built-in sample invoice (no file required)."""
    _banner("DEMO MODE — built-in invoice sample")

    # Pretend Phase 1/2 already ran; feed text directly
    full_text = _DEMO_INVOICE

    # Seed classifier if empty
    if not _classifier.list_types():
        print("[Demo] Seeding classifier with 1 invoice example …")
        _classifier.add_examples("invoice", [full_text])

    classification = _classifier.classify(full_text)
    doc_type   = classification["predicted_type"]
    confidence = classification["confidence"]
    print(f"\n[Phase 3] → '{doc_type}'  confidence={confidence:.3f}")

    extracted  = extract(full_text, doc_type, use_llm=False)
    validation = validate(extracted, doc_type)
    needs_review, reason = should_route_to_review(classification, validation)

    print("\n── DEMO RESULT ──────────────────────────────────────────")
    print(json.dumps(
        {
            "doc_type":   doc_type,
            "confidence": confidence,
            "outcome":    "queued_for_review" if needs_review else "auto_approved",
            "reason":     reason or "high confidence + valid",
            "extracted":  extracted,
            "validation": {
                "passed":   validation["passed"],
                "errors":   validation["errors"],
                "warnings": validation["warnings"],
            },
        },
        indent=2,
        default=str,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(tag: str, msg: str):
    print(f"  [{tag}] {msg}")

def _banner(msg: str):
    print(f"\n{'─'*60}\n{msg}\n{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--demo":
        run_demo()
        sys.exit(0)

    if sys.argv[1] == "--stats":
        print(json.dumps(get_stats(), indent=2))
        sys.exit(0)

    file_arg = sys.argv[1]
    result   = process_document(file_arg)

    print("\n── Final Result ─────────────────────────────────────────")
    print(json.dumps(result, indent=2, default=str))
