"""
Phase 4 — Information Extraction / NER
Tiered approach (as recommended in the POC guide):

  Structured forms  (invoice, expense)  → rule-based regex  [fast, free]
  Free-form docs    (contract, letter)  → LLM via Gemini    [flexible, costs tokens]
  LayoutLMv3                            → STUB              [activate post-POC, needs 500+ labelled docs]

Set your Gemini API key:
    export GEMINI_API_KEY=your-key-here
"""

import json
import os
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM extraction (Claude)
# ─────────────────────────────────────────────────────────────────────────────

# JSON schemas telling the LLM exactly what fields to return per doc type.
_SCHEMAS: dict[str, dict] = {
    "invoice": {
        "vendor_name":    "string",
        "invoice_number": "string",
        "invoice_date":   "YYYY-MM-DD",
        "due_date":       "YYYY-MM-DD",
        "total_amount":   "number (no currency symbol)",
        "tax_amount":     "number (no currency symbol)",
        "currency":       "e.g. INR, USD, EUR",
        "gstin":          "string or null",
        "line_items":     "[{description, quantity, unit_price, total}]",
    },
    "contract": {
        "party_a":        "string",
        "party_b":        "string",
        "contract_date":  "YYYY-MM-DD",
        "effective_date": "YYYY-MM-DD",
        "expiry_date":    "YYYY-MM-DD or null",
        "contract_value": "number or null",
        "jurisdiction":   "string or null",
        "key_clauses":    "[string]",
    },
    "hr_letter": {
        "employee_name": "string",
        "designation":   "string",
        "department":    "string or null",
        "date":          "YYYY-MM-DD",
        "letter_type":   "offer | appraisal | termination | other",
        "ctc":           "number or null",
    },
    "bank_statement": {
        "account_holder": "string",
        "account_number": "string (last 4 digits only)",
        "bank_name":      "string",
        "statement_period_from": "YYYY-MM-DD",
        "statement_period_to":   "YYYY-MM-DD",
        "opening_balance": "number",
        "closing_balance": "number",
        "transactions":    "[{date, description, debit, credit, balance}]",
    },

    # ── Healthcare types ──────────────────────────────────────────────────────
    "patient_registration": {
        "patient_name":      "string",
        "patient_id":        "string or null",
        "dob":               "YYYY-MM-DD",
        "gender":            "Male | Female | Other",
        "blood_group":       "string or null",
        "contact_number":    "string",
        "address":           "string or null",
        "emergency_contact": "string or null",
        "insurance_provider": "string or null",
        "policy_number":     "string or null",
        "registration_date": "YYYY-MM-DD",
    },
    "appointment_letter": {
        "patient_name":    "string",
        "patient_id":      "string or null",
        "doctor_name":     "string",
        "department":      "string",
        "appointment_date": "YYYY-MM-DD",
        "appointment_time": "HH:MM (24h) or null",
        "hospital_name":   "string",
        "reason_for_visit": "string or null",
        "instructions":    "[string] or null",
    },
    "discharge_summary": {
        "patient_name":      "string",
        "patient_id":        "string or null",
        "admission_date":    "YYYY-MM-DD",
        "discharge_date":    "YYYY-MM-DD",
        "attending_physician": "string",
        "diagnosis":         "string",
        "treatment_summary": "string or null",
        "prescribed_medications": "[{name, dosage, frequency}]",
        "follow_up_date":    "YYYY-MM-DD or null",
        "hospital_name":     "string",
    },
    "lab_report": {
        "patient_name":    "string",
        "patient_id":      "string or null",
        "test_name":       "string",
        "sample_collected_date": "YYYY-MM-DD",
        "report_date":     "YYYY-MM-DD",
        "lab_technician":  "string or null",
        "ordered_by":      "string",
        "results":         "[{parameter, value, reference_range, flag}]",
        "overall_result":  "Normal | Abnormal | Inconclusive",
        "hospital_name":   "string or null",
    },
    "medical_invoice": {
        "hospital_name":   "string",
        "gstin":           "string or null",
        "invoice_number":  "string",
        "invoice_date":    "YYYY-MM-DD",
        "patient_name":    "string",
        "patient_id":      "string or null",
        "line_items":      "[{description, quantity, unit_price, total}]",
        "subtotal":        "number",
        "tax_amount":      "number or null",
        "total_amount":    "number",
        "payment_status":  "Paid | Unpaid | Partial",
        "currency":        "e.g. INR",
    },
}


def extract_with_llm(
    text: str,
    doc_type: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Use Gemini to extract structured fields from a document.
    Requires GEMINI_API_KEY env variable (or pass api_key directly).
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("[Phase 4] Run: pip install google-generativeai")
        return {"error": "google_generativeai_not_installed", "_doc_type": doc_type}

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("[Phase 4] GEMINI_API_KEY not set — falling back to rule-based.")
        return _rule_based_fallback(text, doc_type)

    schema = _SCHEMAS.get(doc_type, {"fields": "Extract all key-value pairs from this document."})

    prompt = f"""Extract structured information from the {doc_type} document below.

Return ONLY a valid JSON object that matches this schema exactly:
{json.dumps(schema, indent=2)}

Rules:
- Use null for any field not found in the document
- All dates must be YYYY-MM-DD format
- All monetary amounts must be plain numbers (no ₹ $ £ symbols)
- Do not add extra fields beyond the schema
- Do not include markdown code fences in your response

Document text:
---
{text[:3500]}
---"""

    genai.configure(api_key=key)
    model  = genai.GenerativeModel("gemini-3.5-flash")
    response = model.generate_content(prompt)

    raw = response.text.strip()
    # Strip markdown fences if the model added them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Phase 4] LLM returned non-JSON:\n{raw[:300]}")
        return {"error": "json_parse_failed", "raw": raw, "_doc_type": doc_type}

    extracted["_extraction_method"] = "llm_gemini"
    extracted["_doc_type"] = doc_type
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rule-based extraction (regex, for structured/fixed-layout docs)
# ─────────────────────────────────────────────────────────────────────────────

def _re(pattern: str, text: str, flags=re.IGNORECASE) -> Optional[str]:
    """Return first capture group or None."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _extract_invoice_rules(text: str) -> dict:
    return {
        "vendor_name":    _re(r"^([A-Z][A-Za-z0-9\s&.,\-]+(?:Ltd|Pvt|Inc|Corp|LLP|LLC)?)", text, re.MULTILINE),
        "invoice_number": _re(r"(?:invoice\s*(?:no\.?|#|number)[:\s]*)([A-Z0-9\-/]+)", text),
        "invoice_date":   _re(r"(?:invoice\s*date|date\s*of\s*invoice)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text),
        "due_date":       _re(r"(?:due\s*date|payment\s*due)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text),
        "total_amount":   _re(r"(?:total\s*amount|grand\s*total|amount\s*due|total)[:\s]*[₹$£€]?\s*([\d,]+(?:\.\d{1,2})?)", text),
        "tax_amount":     _re(r"(?:tax|gst|vat|igst|cgst|sgst)[:\s]*[₹$£€]?\s*([\d,]+(?:\.\d{1,2})?)", text),
        "gstin":          _re(r"GSTIN?[:\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z])", text),
        "currency":       _re(r"\b(INR|USD|EUR|GBP|₹)\b", text),
    }


def _extract_expense_rules(text: str) -> dict:
    return {
        "employee_name": _re(r"(?:employee\s*name|name)[:\s]*([A-Za-z\s]+)", text),
        "total_amount":  _re(r"(?:total|amount)[:\s]*[₹$£€]?\s*([\d,]+(?:\.\d{1,2})?)", text),
        "date":          _re(r"(?:date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text),
        "purpose":       _re(r"(?:purpose|description|reason)[:\s]*(.+)", text),
    }


def _extract_patient_registration_rules(text: str) -> dict:
    return {
        "patient_name":      _re(r"(?:patient\s*name|name)[:\s]*([A-Za-z\s]+)", text),
        "patient_id":        _re(r"(?:patient\s*id|pid|uhid)[:\s]*([A-Z0-9\-]+)", text),
        "dob":               _re(r"(?:dob|date\s*of\s*birth|d\.o\.b)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text),
        "gender":            _re(r"(?:gender|sex)[:\s]*(male|female|other)", text, re.IGNORECASE),
        "blood_group":       _re(r"(?:blood\s*group|blood\s*type)[:\s]*([ABO]{1,2}[+-])", text),
        "contact_number":    _re(r"(?:contact|phone|mobile|tel)[:\s]*([+\d\s\-]{10,15})", text),
        "insurance_provider": _re(r"(?:insurance\s*provider|insurer|tpa)[:\s]*([A-Za-z\s]+)", text),
        "policy_number":     _re(r"(?:policy\s*(?:no|number)|policy#)[:\s]*([A-Z0-9\-/]+)", text),
        "registration_date": _re(r"(?:registration\s*date|reg\s*date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text),
    }


def _extract_medical_invoice_rules(text: str) -> dict:
    return {
        "hospital_name":  _re(r"^([A-Z][A-Za-z0-9\s&.,\-]+(?:Hospital|Clinic|Healthcare|Medical Centre)?)", text, re.MULTILINE),
        "invoice_number": _re(r"(?:invoice\s*(?:no\.?|#|number)|bill\s*no)[:\s]*([A-Z0-9\-/]+)", text),
        "invoice_date":   _re(r"(?:invoice\s*date|bill\s*date|date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text),
        "patient_name":   _re(r"(?:patient\s*name|patient)[:\s]*([A-Za-z\s]+)", text),
        "patient_id":     _re(r"(?:patient\s*id|pid|uhid)[:\s]*([A-Z0-9\-]+)", text),
        "total_amount":   _re(r"(?:total\s*amount|grand\s*total|net\s*payable|total)[:\s]*[₹$£€]?\s*([\d,]+(?:\.\d{1,2})?)", text),
        "tax_amount":     _re(r"(?:tax|gst|cgst|sgst)[:\s]*[₹$£€]?\s*([\d,]+(?:\.\d{1,2})?)", text),
        "gstin":          _re(r"GSTIN?[:\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z])", text),
        "payment_status": _re(r"(?:payment\s*status|status)[:\s]*(paid|unpaid|partial)", text, re.IGNORECASE),
        "currency":       _re(r"\b(INR|USD|EUR|GBP|₹)\b", text),
    }


def extract_structured(text: str, doc_type: str) -> dict:
    """Rule-based extraction for well-structured, fixed-layout documents."""
    extractors = {
        "invoice":               _extract_invoice_rules,
        "expense_report":        _extract_expense_rules,
        "patient_registration":  _extract_patient_registration_rules,
        "medical_invoice":       _extract_medical_invoice_rules,
    }
    fn = extractors.get(doc_type)
    if fn is None:
        return _rule_based_fallback(text, doc_type)

    result = fn(text)
    result["_extraction_method"] = "rule_based"
    result["_doc_type"] = doc_type

    filled = sum(1 for k, v in result.items() if v and not k.startswith("_"))
    print(f"[Phase 4] Rule-based: {filled} field(s) extracted for '{doc_type}'")
    return result


def _rule_based_fallback(text: str, doc_type: str) -> dict:
    print(f"[Phase 4] No rule extractor for '{doc_type}' — returning text preview.")
    return {
        "raw_text_preview":   text[:500],
        "warning":            f"No specific extractor for '{doc_type}'. Add rules or enable LLM.",
        "_extraction_method": "fallback",
        "_doc_type":          doc_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. LayoutLMv3 — STUB (activate after 500+ labelled docs per type)
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_layoutlmv3(
    text: str,
    words_with_bbox: list,
    doc_type: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    LayoutLMv3-based extraction. STUB — not active in POC.

    To activate later:
      1. Label documents in Label Studio (export as FUNSD/CORD JSON)
      2. Fine-tune LayoutLMv3 using Hugging Face Trainer
         → https://huggingface.co/docs/transformers/model_doc/layoutlmv3
      3. Save the model to  models/layoutlmv3_<doc_type>/
      4. Replace the NotImplementedError below with inference code

    M1 Mac note: set device="mps" once MPS is stable for LayoutLMv3.
    """
    model_path = f"models/layoutlmv3_{doc_type}"

    if not os.path.exists(model_path):
        print(f"[Phase 4] LayoutLMv3 not found at {model_path} — falling back to Gemini LLM.")
        return extract_with_llm(text, doc_type, api_key)

    # ── TODO: real inference (uncomment after training) ──────────────────────
    # from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    # processor = LayoutLMv3Processor.from_pretrained(model_path)
    # model    = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    # ...
    raise NotImplementedError(
        "LayoutLMv3 training not complete. "
        "See docstring for steps, or use extract_with_llm() instead."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main router (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

# Doc types that have good rule-based coverage.
_STRUCTURED_TYPES = {"invoice", "expense_report", "patient_registration", "medical_invoice"}

# Minimum number of non-null fields for rule-based result to be accepted.
_MIN_FIELDS = 3


def extract(
    text: str,
    doc_type: str,
    words_with_bbox: list = None,
    use_llm: bool = True,
    api_key: Optional[str] = None,
) -> dict:
    """
    Main extraction entry point.

    Logic
    -----
    1. Structured types  → try rule-based first
       • If < MIN_FIELDS filled and use_llm=True → upgrade to LLM
    2. Everything else   → LLM directly (or fallback if no API key)
    """
    if doc_type in _STRUCTURED_TYPES:
        result = extract_structured(text, doc_type)
        filled = sum(
            1 for k, v in result.items()
            if v is not None and not k.startswith("_")
        )
        if filled < _MIN_FIELDS and use_llm:
            print(
                f"[Phase 4] Only {filled} fields from rules "
                f"(need {_MIN_FIELDS}) — upgrading to LLM."
            )
            return extract_with_llm(text, doc_type, api_key)
        return result

    if use_llm:
        return extract_with_llm(text, doc_type, api_key)

    return _rule_based_fallback(text, doc_type)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sample_invoice = """
    ACME Supplies Pvt Ltd
    123 Industrial Area, Mumbai

    Invoice No: INV-2024-0042
    Invoice Date: 15/07/2024
    Due Date: 30/07/2024
    GSTIN: 27AAAPL1234C1Z5

    Item          Qty   Rate      Total
    Widget A       10   500.00    5000.00
    Widget B        5  1000.00    5000.00

    Subtotal:  10000.00
    GST (18%):  1800.00
    Total:     11800.00  INR
    """

    result = extract(sample_invoice, "invoice", use_llm=False)
    print(json.dumps(result, indent=2))
