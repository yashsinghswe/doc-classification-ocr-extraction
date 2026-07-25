"""
Phase 1 — OCR Layer (EasyOCR)
Converts scanned images / non-searchable PDFs into text + bounding boxes.
Skip this phase entirely if the document is a digital PDF → use Phase 2 instead.

M1 Mac note: EasyOCR's MPS backend is experimental; gpu=False is intentional here.
"""

import numpy as np
from pathlib import Path


# ── Detect if a PDF is scanned (no embedded text) ────────────────────────────

def is_scanned_pdf(pdf_path: str, text_threshold: int = 20) -> bool:
    """
    Returns True  → scanned, needs OCR (Phase 1)
    Returns False → digital, go to Phase 2
    """
    import fitz  # pymupdf
    doc = fitz.open(pdf_path)
    total_text = "".join(page.get_text() for page in doc)
    doc.close()
    return len(total_text.strip()) < text_threshold


# ── Convert PDF pages to PIL images ──────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 300):
    """Rasterize each PDF page to a PIL Image at the given DPI."""
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"[Phase 1] Rasterized {len(images)} page(s) at {dpi} dpi")
    return images


# ── Core EasyOCR runner ───────────────────────────────────────────────────────

def run_easyocr(
    image_path: str = None,
    pdf_path: str = None,
    languages: list = ["en"],
    dpi: int = 300,
) -> dict:
    """
    Run EasyOCR on an image file or a scanned PDF.

    Returns
    -------
    {
        "source":          "easyocr",
        "full_text":       str,
        "words":           [{"text", "confidence", "page", "bbox": {x,y,w,h}}],
        "avg_confidence":  float,
        "page_count":      int,
    }
    """
    import easyocr

    if image_path is None and pdf_path is None:
        raise ValueError("Provide either image_path or pdf_path")

    # Initialise reader once (model downloads on first run, ~200 MB)
    # gpu=False: safe default for M1 Mac (MPS support is experimental in EasyOCR)
    reader = easyocr.Reader(languages, gpu=False)
    print(f"[Phase 1] EasyOCR reader ready (languages: {languages})")

    all_words = []

    if pdf_path:
        images = pdf_to_images(pdf_path, dpi=dpi)
        for page_num, pil_image in enumerate(images, start=1):
            img_array = np.array(pil_image)
            raw = reader.readtext(img_array)
            all_words.extend(_parse_results(raw, page_num))
            print(f"[Phase 1]   Page {page_num}: {len(raw)} text regions found")

    else:
        raw = reader.readtext(image_path)
        all_words = _parse_results(raw, page_num=1)
        print(f"[Phase 1] {len(raw)} text regions found in image")

    full_text = " ".join(w["text"] for w in all_words)
    avg_conf = float(np.mean([w["confidence"] for w in all_words])) if all_words else 0.0
    page_count = len(set(w["page"] for w in all_words))

    return {
        "source": "easyocr",
        "full_text": full_text,
        "words": all_words,
        "avg_confidence": round(avg_conf, 3),
        "page_count": page_count,
    }


# ── Parse raw EasyOCR output ──────────────────────────────────────────────────

def _parse_results(raw_results: list, page_num: int) -> list:
    """
    raw_results item: (bbox, text, confidence)
    bbox: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]  (four corners)
    """
    parsed = []
    for bbox, text, confidence in raw_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        parsed.append(
            {
                "text": text,
                "confidence": round(float(confidence), 3),
                "page": page_num,
                "bbox": {
                    "x": round(min(xs), 1),
                    "y": round(min(ys), 1),
                    "width": round(max(xs) - min(xs), 1),
                    "height": round(max(ys) - min(ys), 1),
                },
            }
        )
    return parsed


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python phase1_ocr.py <image_or_pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    if path.endswith(".pdf"):
        result = run_easyocr(pdf_path=path)
    else:
        result = run_easyocr(image_path=path)

    print(f"\nText preview (first 400 chars):\n{result['full_text'][:400]}")
    print(f"\nAvg confidence : {result['avg_confidence']}")
    print(f"Pages processed: {result['page_count']}")
    print(f"Words found    : {len(result['words'])}")
