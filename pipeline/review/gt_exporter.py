"""
Export a completed review session as a ground truth JSON file.

Matches the format in configs/ground_truth/*.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .review_session import ReviewSession


def export_ground_truth(
    session: ReviewSession,
    output_path: Path | None = None,
) -> Path:
    """
    Convert a reviewed session into ground truth JSON.

    Args:
        session: Completed ReviewSession with decisions
        output_path: Where to write. Defaults to configs/ground_truth/<fund>.json

    Returns:
        Path to written file
    """
    if output_path is None:
        safe_name = session.fund_name.lower().replace(" ", "_").replace("/", "_")[:50]
        output_path = Path("configs/ground_truth") / f"{safe_name}.json"

    gt: dict[str, Any] = {
        "fund_name": session.fund_name,
        "cik": session.cik,
        "filing_date": session.filing_date,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "verified_by": "hitl_review",
        "notes": f"Created via HITL review of experiment {Path(session.experiment_dir).name}",
        "fields": {},
    }

    for field_key, review in session.fields.items():
        if not review.is_reviewed:
            continue

        entry: dict[str, Any] = {}

        if review.decision == "na":
            entry["expected"] = None
            entry["nullable"] = True
            if review.reviewer_notes:
                entry["notes"] = review.reviewer_notes
            else:
                entry["notes"] = "Marked N/A by reviewer"
        elif review.decision == "accept":
            entry["expected"] = _normalize_value(review.extracted_value)
            entry["nullable"] = False
            if review.evidence:
                # Extract section from evidence
                section = _extract_section_from_evidence(review.evidence)
                if section:
                    entry["source_section"] = section
                # Use first 200 chars of evidence as quote
                quote = _extract_quote_from_evidence(review.evidence)
                if quote:
                    entry["source_quote"] = quote
            if review.reviewer_notes:
                entry["notes"] = review.reviewer_notes
            else:
                entry["notes"] = "Accepted from extraction"
        elif review.decision == "correct":
            entry["expected"] = _normalize_value(review.corrected_value)
            entry["nullable"] = False
            if review.evidence:
                section = _extract_section_from_evidence(review.evidence)
                if section:
                    entry["source_section"] = section
                quote = _extract_quote_from_evidence(review.evidence)
                if quote:
                    entry["source_quote"] = quote
            if review.reviewer_notes:
                entry["notes"] = review.reviewer_notes
            else:
                entry["notes"] = f"Corrected from '{review.extracted_value}'"

        gt["fields"][field_key] = entry

    # Handle share class fields: group into share_classes.share_classes format
    _consolidate_share_classes(gt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, default=str)

    return output_path


def _normalize_value(value: Any) -> Any:
    """Normalize extracted value for GT format."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    # Try to preserve numeric strings as-is
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null", ""):
        return None
    return s


def _extract_section_from_evidence(evidence: str) -> str | None:
    """Extract [Section: X] from evidence string."""
    if evidence and evidence.startswith("[Section: "):
        end = evidence.find("]")
        if end > 0:
            return evidence[10:end]
    return None


def _extract_quote_from_evidence(evidence: str) -> str | None:
    """Extract a usable quote from evidence, stripping section prefix."""
    if not evidence:
        return None
    text = evidence
    # Remove section prefix
    if text.startswith("[Section: "):
        end = text.find("]\n")
        if end > 0:
            text = text[end + 2:]
    # Take first 200 chars as quote
    text = text.strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text if text else None


def _consolidate_share_classes(gt: dict) -> None:
    """
    Convert individual share_classes.ClassName.field entries into
    the share_classes.share_classes list format used by existing GT files.
    """
    sc_fields: dict[str, dict] = {}  # class_name -> {field: value}
    keys_to_remove = []

    for field_key, entry in list(gt["fields"].items()):
        parts = field_key.split(".")
        if len(parts) == 3 and parts[0] == "share_classes":
            class_name = parts[1]
            sc_field = parts[2]
            if class_name not in sc_fields:
                sc_fields[class_name] = {"class_name": class_name}
            sc_fields[class_name][sc_field] = entry.get("expected")
            keys_to_remove.append(field_key)

    if sc_fields:
        for k in keys_to_remove:
            del gt["fields"][k]
        gt["fields"]["share_classes.share_classes"] = {
            "expected": list(sc_fields.values()),
            "match_mode": "exact",
            "nullable": False,
        }
