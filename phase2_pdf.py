"""
Phase 2 — Digital PDF Text Extraction
Uses pdfplumber (table-heavy docs) and pymupdf (text-heavy / high-volume docs).
Only runs when the PDF has embedded text; skip to Phase 1 for scans.
"""

from pathlib import Path


# ── Is the PDF digital or scanned? ───────────────────────────────────────────

def is_digital_pdf(pdf_path: str, text_threshold: int = 20) -> bool:
    """
    Returns True if the PDF has embedded text (digital).
    Returns False if it's a scanned image PDF → use Phase 1.
    """
    import fitz
    doc = fitz.open(pdf_path)
    total = "".join(page.get_text() for page in doc)
    doc.close()
    return len(total.strip()) >= text_threshold


# ── pdfplumber extraction (best for tables) ───────────────────────────────────

def extract_with_pdfplumber(pdf_path: str) -> dict:
    """
    Extracts text, tables, and word bounding boxes using pdfplumber.
    Best choice: invoices, financial statements, any doc with structured tables.
    """
    import pdfplumber

    full_text = ""
    tables = []
    words = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):

            # Plain text
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"

            # Tables → list of lists (rows × cols)
            for tbl in page.extract_tables():
                tables.append({"page": page_num, "data": tbl})

            # Word-level bounding boxes
            for w in page.extract_words():
                words.append(
                    {
                        "text": w["text"],
                        "page": page_num,
                        "bbox": {
                            "x": round(w["x0"], 1),
                            "y": round(w["top"], 1),
                            "width": round(w["x1"] - w["x0"], 1),
                            "height": round(w["bottom"] - w["top"], 1),
                        },
                    }
                )

    print(
        f"[Phase 2 / pdfplumber] {total_pages}p · "
        f"{len(full_text.split())} words · {len(tables)} table(s)"
    )

    return {
        "source": "pdfplumber",
        "full_text": full_text.strip(),
        "tables": tables,
        "words": words,
        "page_count": total_pages,
    }


# ── pymupdf extraction (best for speed / plain text) ─────────────────────────

def extract_with_pymupdf(pdf_path: str) -> dict:
    """
    Extracts text and word bounding boxes using PyMuPDF (fitz).
    Best choice: contracts, letters, plain-text documents at high volume.
    10–50× faster than pdfplumber; no native table support.
    """
    import fitz

    doc = fitz.open(pdf_path)
    full_text = ""
    words = []

    for page_num, page in enumerate(doc, start=1):
        full_text += page.get_text() + "\n"

        # get_text("words") → (x0, y0, x1, y1, word, block, line, word_idx)
        for w in page.get_text("words"):
            words.append(
                {
                    "text": w[4],
                    "page": page_num,
                    "bbox": {
                        "x": round(w[0], 1),
                        "y": round(w[1], 1),
                        "width": round(w[2] - w[0], 1),
                        "height": round(w[3] - w[1], 1),
                    },
                }
            )

    page_count = len(doc)
    doc.close()

    print(
        f"[Phase 2 / pymupdf] {page_count}p · "
        f"{len(full_text.split())} words"
    )

    return {
        "source": "pymupdf",
        "full_text": full_text.strip(),
        "tables": [],   # pymupdf has no native table extractor
        "words": words,
        "page_count": page_count,
    }


# ── Smart router (called by main.py) ─────────────────────────────────────────

def extract_digital_pdf(pdf_path: str) -> dict:
    """
    Auto-selects the right extractor:
    • pdfplumber first (to find tables)
    • falls back to pymupdf if no tables found (faster)
    • merges table data from pdfplumber into pymupdf result when both are run
    """
    plumber_result = extract_with_pdfplumber(pdf_path)

    if plumber_result["tables"]:
        # Has tables → pdfplumber result is authoritative
        return plumber_result

    # No tables found → use pymupdf for speed, but keep same schema
    mupdf_result = extract_with_pymupdf(pdf_path)
    mupdf_result["tables"] = []   # already empty, but explicit
    return mupdf_result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python phase2_pdf.py <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]

    if not is_digital_pdf(path):
        print("⚠  PDF appears to be scanned — use Phase 1 (EasyOCR) instead.")
        sys.exit(0)

    result = extract_digital_pdf(path)

    print(f"\nText preview (first 400 chars):\n{result['full_text'][:400]}")
    print(f"\nPages   : {result['page_count']}")
    print(f"Tables  : {len(result['tables'])}")
    print(f"Words   : {len(result['words'])}")

    if result["tables"]:
        print(f"\nFirst table (first 3 rows):")
        for row in result["tables"][0]["data"][:3]:
            print(" | ".join(str(c) for c in row))
