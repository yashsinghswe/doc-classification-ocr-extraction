"""
Phase 6 — Human-in-the-Loop Review Queue
Routes low-confidence documents to a review queue and feeds corrections
back into the Phase 3 embedding index (the self-improving loop).

Storage: SQLite via SQLAlchemy (no server needed, single file on disk).
Review UI: Flask app in review_ui/app.py
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH            = "models/review_queue.db"
CONFIDENCE_THRESH  = 0.75   # documents below this go to review


# ── ORM model ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class ReviewItem(Base):
    __tablename__ = "review_queue"

    id         = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Source document
    file_path  = Column(String)
    full_text  = Column(Text)          # truncated to 5000 chars for storage

    # Phase 3 output
    predicted_type             = Column(String)
    classification_confidence  = Column(Float)
    all_scores_json            = Column(Text)  # JSON dict

    # Phase 4 output
    extracted_json  = Column(Text)  # JSON

    # Phase 5 output
    validation_passed  = Column(Boolean)
    validation_errors  = Column(Text)   # JSON list
    validation_warnings = Column(Text)  # JSON list

    # Routing
    route_reason = Column(String)   # "low_confidence" | "validation_failed" | "both"

    # Review outcome
    reviewed       = Column(Boolean,  default=False)
    confirmed_type = Column(String)
    corrected_json = Column(Text)
    reviewed_at    = Column(DateTime)
    reviewer_note  = Column(Text)


# ── DB init (idempotent) ──────────────────────────────────────────────────────

def _get_engine():
    os.makedirs("models", exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db():
    engine = _get_engine()
    Base.metadata.create_all(engine)
    return engine


# ── Routing decision ──────────────────────────────────────────────────────────

def should_route_to_review(
    classification: dict,
    validation: dict,
    threshold: float = CONFIDENCE_THRESH,
) -> tuple[bool, str]:
    """
    Returns (needs_review: bool, reason: str).

    Triggers review when:
      • Classification confidence < threshold   (model is uncertain)
      • Validation has hard errors              (data is corrupt / implausible)
      • Both of the above
    """
    low_conf  = classification.get("confidence", 0.0) < threshold
    val_fail  = not validation.get("passed", True)

    if low_conf and val_fail:
        conf = classification["confidence"]
        errs = "; ".join(validation.get("errors", [])[:2])
        return True, f"low_confidence ({conf:.2f}) + validation_failed: {errs}"

    if low_conf:
        conf = classification["confidence"]
        return True, f"low_confidence ({conf:.2f} < {threshold})"

    if val_fail:
        errs = "; ".join(validation.get("errors", [])[:2])
        return True, f"validation_failed: {errs}"

    return False, ""


# ── Queue operations ──────────────────────────────────────────────────────────

def add_to_queue(
    file_path: str,
    full_text: str,
    classification: dict,
    extraction: dict,
    validation: dict,
    route_reason: str,
) -> str:
    """
    Add a document to the review queue.
    Returns the UUID of the created item.
    """
    engine = init_db()

    item = ReviewItem(
        file_path                  = str(file_path),
        full_text                  = full_text[:5000],
        predicted_type             = classification.get("predicted_type"),
        classification_confidence  = classification.get("confidence"),
        all_scores_json            = json.dumps(classification.get("all_scores", {})),
        extracted_json             = json.dumps(extraction, default=str),
        validation_passed          = validation.get("passed"),
        validation_errors          = json.dumps(validation.get("errors", [])),
        validation_warnings        = json.dumps(validation.get("warnings", [])),
        route_reason               = route_reason,
    )

    with Session(engine) as session:
        session.add(item)
        session.commit()
        item_id = item.id

    print(f"[Phase 6] Queued for review → id={item_id[:8]}… reason='{route_reason}'")
    return item_id


def confirm_review(
    item_id: str,
    confirmed_type: str,
    corrected_data: dict,
    note: str = "",
) -> bool:
    """
    Record a human reviewer's decision.
    Automatically feeds the correction back into Phase 3's embedding index.

    Parameters
    ----------
    item_id        : UUID from add_to_queue()
    confirmed_type : The correct document type (may differ from predicted_type)
    corrected_data : The corrected extraction JSON
    note           : Optional reviewer comment
    """
    engine = init_db()

    with Session(engine) as session:
        item = session.get(ReviewItem, item_id)
        if not item:
            print(f"[Phase 6] Item {item_id} not found in queue.")
            return False

        item.reviewed       = True
        item.confirmed_type = confirmed_type
        item.corrected_json = json.dumps(corrected_data, default=str)
        item.reviewed_at    = datetime.utcnow()
        item.reviewer_note  = note
        session.commit()

        full_text = item.full_text or ""

    # ── Feed back to Phase 3 (the self-improving loop) ────────────────────────
    try:
        from phase3_classify import FewShotClassifier
        clf = FewShotClassifier()
        clf.add_from_review(confirmed_type, full_text)
        print(f"[Phase 6] Phase 3 centroid updated for '{confirmed_type}'.")
    except Exception as exc:
        print(f"[Phase 6] Could not update Phase 3 index: {exc}")

    print(f"[Phase 6] Review confirmed: {item_id[:8]}… → '{confirmed_type}'")
    return True


def get_pending_items(limit: int = 20) -> list[dict]:
    """Return unreviewed queue items for the review UI."""
    engine = init_db()

    with Session(engine) as session:
        items = (
            session.query(ReviewItem)
            .filter(ReviewItem.reviewed == False)
            .order_by(ReviewItem.created_at)
            .limit(limit)
            .all()
        )
        return [_item_to_dict(item) for item in items]


def get_item(item_id: str) -> Optional[dict]:
    """Fetch a single review item by ID."""
    engine = init_db()
    with Session(engine) as session:
        item = session.get(ReviewItem, item_id)
        return _item_to_dict(item) if item else None


def get_stats() -> dict:
    """Queue statistics for the review UI dashboard."""
    engine = init_db()
    with Session(engine) as session:
        total    = session.query(ReviewItem).count()
        pending  = session.query(ReviewItem).filter(ReviewItem.reviewed == False).count()
        reviewed = total - pending
        return {
            "total":        total,
            "pending":      pending,
            "reviewed":     reviewed,
            "pending_rate": f"{(pending / total * 100):.1f}%" if total else "N/A",
            "review_completion_rate": f"{(reviewed / total * 100):.1f}%" if total else "N/A",
        }


def _item_to_dict(item: ReviewItem) -> dict:
    return {
        "id":           item.id,
        "created_at":   str(item.created_at),
        "file_path":    item.file_path,
        "text_preview": (item.full_text or "")[:400],
        "predicted_type":   item.predicted_type,
        "confidence":       item.classification_confidence,
        "all_scores":       json.loads(item.all_scores_json or "{}"),
        "extracted":        json.loads(item.extracted_json or "{}"),
        "validation_passed": item.validation_passed,
        "errors":       json.loads(item.validation_errors  or "[]"),
        "warnings":     json.loads(item.validation_warnings or "[]"),
        "route_reason": item.route_reason,
        "reviewed":     item.reviewed,
        "confirmed_type": item.confirmed_type,
        "corrected":    json.loads(item.corrected_json or "{}") if item.corrected_json else {},
        "reviewer_note": item.reviewer_note,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    stats = get_stats()
    print("Queue stats:", json.dumps(stats, indent=2))

    pending = get_pending_items()
    print(f"Pending items: {len(pending)}")
    for p in pending[:3]:
        print(f"  • {p['id'][:8]} | {p['predicted_type']} ({p['confidence']:.2f}) | {p['route_reason']}")
