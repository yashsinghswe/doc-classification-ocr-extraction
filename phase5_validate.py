"""
Phase 5 — Post-Processing & Validation
Catches OCR errors, format mismatches, and implausible values before they
flow into downstream systems.

Three layers (as recommended in the POC guide):
  1. Pydantic schema validation — type enforcement, required fields
  2. Regex format checks        — dates, GSTIN, PAN, phone, amounts
  3. Cross-field arithmetic     — line_items_sum == total_amount ± ₹1
"""

import re
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class InvoiceSchema(BaseModel):
    vendor_name:    Optional[str]   = None
    invoice_number: Optional[str]   = None
    invoice_date:   Optional[str]   = None
    due_date:       Optional[str]   = None
    total_amount:   Optional[float] = None
    tax_amount:     Optional[float] = None
    currency:       Optional[str]   = None
    gstin:          Optional[str]   = None

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)

    @field_validator("total_amount", "tax_amount", mode="before")
    @classmethod
    def normalise_amount(cls, v):
        return _clean_amount(v)

    @field_validator("gstin", mode="before")
    @classmethod
    def validate_gstin(cls, v):
        if v and not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$", str(v)):
            raise ValueError(f"Invalid GSTIN format: {v}")
        return v


class ContractSchema(BaseModel):
    party_a:        Optional[str]   = None
    party_b:        Optional[str]   = None
    contract_date:  Optional[str]   = None
    effective_date: Optional[str]   = None
    expiry_date:    Optional[str]   = None
    contract_value: Optional[float] = None

    @field_validator("contract_date", "effective_date", "expiry_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)

    @field_validator("contract_value", mode="before")
    @classmethod
    def normalise_amount(cls, v):
        return _clean_amount(v)


class HRLetterSchema(BaseModel):
    employee_name: Optional[str]   = None
    designation:   Optional[str]   = None
    department:    Optional[str]   = None
    date:          Optional[str]   = None
    letter_type:   Optional[str]   = None
    ctc:           Optional[float] = None

    @field_validator("date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)

    @field_validator("ctc", mode="before")
    @classmethod
    def normalise_ctc(cls, v):
        return _clean_amount(v)


class PatientRegistrationSchema(BaseModel):
    patient_name:      Optional[str]   = None
    patient_id:        Optional[str]   = None
    dob:               Optional[str]   = None
    gender:            Optional[str]   = None
    blood_group:       Optional[str]   = None
    contact_number:    Optional[str]   = None
    insurance_provider: Optional[str]  = None
    policy_number:     Optional[str]   = None
    registration_date: Optional[str]   = None

    @field_validator("dob", "registration_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)

    @field_validator("gender", mode="before")
    @classmethod
    def normalise_gender(cls, v):
        if v:
            normalised = str(v).strip().capitalize()
            if normalised not in ("Male", "Female", "Other"):
                raise ValueError(f"Invalid gender value: {v!r}")
            return normalised
        return v


class AppointmentLetterSchema(BaseModel):
    patient_name:     Optional[str] = None
    patient_id:       Optional[str] = None
    doctor_name:      Optional[str] = None
    department:       Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    hospital_name:    Optional[str] = None
    reason_for_visit: Optional[str] = None

    @field_validator("appointment_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)


class DischargeSummarySchema(BaseModel):
    patient_name:        Optional[str] = None
    patient_id:          Optional[str] = None
    admission_date:      Optional[str] = None
    discharge_date:      Optional[str] = None
    attending_physician: Optional[str] = None
    diagnosis:           Optional[str] = None
    follow_up_date:      Optional[str] = None
    hospital_name:       Optional[str] = None

    @field_validator("admission_date", "discharge_date", "follow_up_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)


class LabReportSchema(BaseModel):
    patient_name:         Optional[str] = None
    patient_id:           Optional[str] = None
    test_name:            Optional[str] = None
    sample_collected_date: Optional[str] = None
    report_date:          Optional[str] = None
    ordered_by:           Optional[str] = None
    overall_result:       Optional[str] = None

    @field_validator("sample_collected_date", "report_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)

    @field_validator("overall_result", mode="before")
    @classmethod
    def normalise_result(cls, v):
        if v:
            normalised = str(v).strip().capitalize()
            if normalised not in ("Normal", "Abnormal", "Inconclusive"):
                raise ValueError(f"Invalid overall_result: {v!r}")
            return normalised
        return v


class MedicalInvoiceSchema(BaseModel):
    hospital_name:  Optional[str]   = None
    gstin:          Optional[str]   = None
    invoice_number: Optional[str]   = None
    invoice_date:   Optional[str]   = None
    patient_name:   Optional[str]   = None
    patient_id:     Optional[str]   = None
    subtotal:       Optional[float] = None
    tax_amount:     Optional[float] = None
    total_amount:   Optional[float] = None
    payment_status: Optional[str]   = None
    currency:       Optional[str]   = None

    @field_validator("invoice_date", mode="before")
    @classmethod
    def normalise_date(cls, v):
        return _fix_ocr_date(v)

    @field_validator("subtotal", "tax_amount", "total_amount", mode="before")
    @classmethod
    def normalise_amount(cls, v):
        return _clean_amount(v)

    @field_validator("gstin", mode="before")
    @classmethod
    def validate_gstin(cls, v):
        if v and not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$", str(v)):
            raise ValueError(f"Invalid GSTIN format: {v}")
        return v

    @field_validator("payment_status", mode="before")
    @classmethod
    def normalise_payment_status(cls, v):
        if v:
            normalised = str(v).strip().capitalize()
            if normalised not in ("Paid", "Unpaid", "Partial"):
                raise ValueError(f"Invalid payment_status: {v!r}")
            return normalised
        return v


_SCHEMAS = {
    "invoice":               InvoiceSchema,
    "contract":              ContractSchema,
    "hr_letter":             HRLetterSchema,
    "patient_registration":  PatientRegistrationSchema,
    "appointment_letter":    AppointmentLetterSchema,
    "discharge_summary":     DischargeSummarySchema,
    "lab_report":            LabReportSchema,
    "medical_invoice":       MedicalInvoiceSchema,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"]


def _fix_ocr_date(value) -> Optional[str]:
    """
    Fix common OCR substitutions in dates (O→0, l→1, S→5)
    and normalise to the first parseable format found.
    Returns None if value is None or empty.
    """
    if not value:
        return None
    s = str(value).strip()
    # Common OCR character swaps
    s = re.sub(r"(?<!\w)O(?!\w)", "0", s)   # letter O → zero  (word boundary)
    s = s.replace("l", "1").replace("S", "5")
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return s          # valid — return as-is (don't force YYYY-MM-DD for now)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {value!r}")


def _clean_amount(value) -> Optional[float]:
    """Strip currency symbols and commas, return float or None."""
    if value is None:
        return None
    cleaned = re.sub(r"[₹$£€,\s]", "", str(value)).strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"Cannot parse amount: {value!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Format checkers (beyond Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

def check_pan(value: str) -> bool:
    return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", str(value))) if value else True


def check_phone(value: str) -> bool:
    if not value:
        return True
    digits = re.sub(r"\D", "", str(value))
    return len(digits) in (10, 12)


def check_gstin(value: str) -> bool:
    if not value:
        return True
    return bool(re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$", str(value)))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-field checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_line_items_sum(extracted: dict, tolerance: float = 1.0) -> tuple[bool, str]:
    """Sum of line_item totals should equal total_amount ± tolerance."""
    items = extracted.get("line_items") or []
    total = extracted.get("total_amount")

    if not items or total is None:
        return True, ""

    try:
        items_sum = sum(float(item.get("total") or 0) for item in items)
        diff = abs(items_sum - float(total))
        if diff > tolerance:
            return False, f"Line items sum ({items_sum:.2f}) ≠ total ({float(total):.2f}); diff={diff:.2f}"
        return True, ""
    except (TypeError, ValueError):
        return True, ""   # can't compute → skip


def _check_date_order(
    earlier_field: str,
    later_field: str,
    extracted: dict,
) -> tuple[bool, str]:
    """earlier_field date must be ≤ later_field date."""
    d1_raw = extracted.get(earlier_field)
    d2_raw = extracted.get(later_field)

    if not d1_raw or not d2_raw:
        return True, ""

    for fmt in _DATE_FORMATS:
        try:
            d1 = datetime.strptime(str(d1_raw), fmt)
            d2 = datetime.strptime(str(d2_raw), fmt)
            if d1 > d2:
                return False, (
                    f"'{earlier_field}' ({d1_raw}) is after '{later_field}' ({d2_raw})"
                )
            return True, ""
        except ValueError:
            continue

    return True, ""   # unparseable dates → skip


def _check_future_year(field: str, extracted: dict, max_year: int = 2100) -> tuple[bool, str]:
    """Catch obviously wrong years (e.g., 2O31 → 2031 after OCR fix, but still implausible)."""
    raw = extracted.get(field)
    if not raw:
        return True, ""
    year_match = re.search(r"\b(20\d{2})\b", str(raw))
    if year_match:
        year = int(year_match.group(1))
        if year > datetime.now().year + 2:
            return False, f"'{field}' year {year} looks implausible"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main validate() function
# ─────────────────────────────────────────────────────────────────────────────

def validate(extracted: dict, doc_type: str) -> dict:
    """
    Run all three validation layers on extracted fields.

    Returns
    -------
    {
        "passed":       bool,
        "errors":       [str],   # hard failures → route to human review
        "warnings":     [str],   # soft issues → log but don't block
        "cleaned_data": dict,    # normalised copy of extracted
        "needs_review": bool,
    }
    """
    errors: list[str]   = []
    warnings: list[str] = []

    # Work on a clean copy; preserve private _keys
    cleaned = {k: v for k, v in extracted.items()}

    # ── Layer 1: Pydantic ────────────────────────────────────────────────────
    schema_cls = _SCHEMAS.get(doc_type)
    if schema_cls:
        # Only pass public fields (Pydantic ignores extras by default)
        public = {k: v for k, v in extracted.items() if not k.startswith("_")}
        try:
            validated_obj = schema_cls(**public)
            cleaned.update(validated_obj.model_dump())
        except Exception as exc:
            for err in str(exc).split("\n"):
                if err.strip():
                    errors.append(f"Schema: {err.strip()}")

    # ── Layer 2: Extra regex checks ──────────────────────────────────────────
    if cleaned.get("pan") and not check_pan(cleaned["pan"]):
        errors.append(f"Invalid PAN format: {cleaned['pan']}")

    if cleaned.get("phone") and not check_phone(cleaned["phone"]):
        warnings.append(f"Unexpected phone length: {cleaned['phone']}")

    if cleaned.get("gstin") and not check_gstin(cleaned["gstin"]):
        errors.append(f"Invalid GSTIN: {cleaned['gstin']}")

    # Date plausibility (all date fields)
    for date_field in [
        "invoice_date", "due_date", "contract_date", "effective_date", "expiry_date", "date",
        "dob", "registration_date", "appointment_date",
        "admission_date", "discharge_date", "follow_up_date",
        "sample_collected_date", "report_date",
    ]:
        ok, msg = _check_future_year(date_field, cleaned)
        if not ok:
            warnings.append(msg)

    # ── Layer 3: Cross-field checks ──────────────────────────────────────────
    if doc_type == "invoice":
        ok, msg = _check_line_items_sum(extracted)   # use original (floats may differ after cleaning)
        if not ok:
            warnings.append(msg)

        ok, msg = _check_date_order("invoice_date", "due_date", cleaned)
        if not ok:
            errors.append(msg)

    if doc_type == "contract":
        ok, msg = _check_date_order("effective_date", "expiry_date", cleaned)
        if not ok:
            errors.append(msg)

    if doc_type == "discharge_summary":
        ok, msg = _check_date_order("admission_date", "discharge_date", cleaned)
        if not ok:
            errors.append(msg)

        ok, msg = _check_date_order("discharge_date", "follow_up_date", cleaned)
        if not ok:
            errors.append(msg)

    if doc_type == "lab_report":
        ok, msg = _check_date_order("sample_collected_date", "report_date", cleaned)
        if not ok:
            errors.append(msg)

    if doc_type == "medical_invoice":
        ok, msg = _check_line_items_sum(extracted)
        if not ok:
            warnings.append(msg)

        if cleaned.get("gstin") and not check_gstin(cleaned["gstin"]):
            errors.append(f"Invalid GSTIN: {cleaned['gstin']}")

        subtotal = cleaned.get("subtotal")
        tax      = cleaned.get("tax_amount") or 0
        total    = cleaned.get("total_amount")
        if subtotal is not None and total is not None:
            expected = float(subtotal) + float(tax)
            if abs(expected - float(total)) > 1.0:
                warnings.append(
                    f"subtotal ({subtotal}) + tax ({tax}) = {expected:.2f} "
                    f"≠ total_amount ({total})"
                )

    passed = len(errors) == 0

    print(
        f"[Phase 5] Validation {'✅ PASSED' if passed else '❌ FAILED'} "
        f"({len(errors)} errors, {len(warnings)} warnings)"
    )
    for e in errors:
        print(f"  ERROR   : {e}")
    for w in warnings:
        print(f"  WARNING : {w}")

    return {
        "passed":       passed,
        "errors":       errors,
        "warnings":     warnings,
        "cleaned_data": cleaned,
        "needs_review": not passed,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample = {
        "vendor_name":    "ACME Supplies Pvt Ltd",
        "invoice_number": "INV-2024-0042",
        "invoice_date":   "15/07/2024",
        "due_date":       "3O/07/2024",   # OCR error: O instead of 0
        "total_amount":   "11,800.00",
        "tax_amount":     "1800",
        "gstin":          "27AAAPL1234C1Z5",
        "currency":       "INR",
        "_extraction_method": "rule_based",
        "_doc_type": "invoice",
    }

    result = validate(sample, "invoice")
    print("\n── Validation result ──")
    print(json.dumps(result, indent=2, default=str))
