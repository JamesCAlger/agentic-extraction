"""
Review session persistence for HITL ground truth creation.

Saves/resumes review progress so a reviewer can stop and continue later.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class FieldReview:
    """Review decision for a single field."""
    field_key: str  # e.g. "incentive_fee.hurdle_rate_pct"
    category: str  # e.g. "Incentive Fee"
    extracted_value: Any
    evidence: Optional[str] = None
    section: Optional[str] = None
    confidence: Optional[str] = None  # from trace: "explicit", "inferred", etc.
    grounded: Optional[bool] = None
    tier: Optional[str] = None  # "t3", "reranker", "agreement", etc.
    reasoning: Optional[str] = None

    # Review decision
    decision: Optional[str] = None  # "accept", "correct", "na"
    corrected_value: Any = None
    reviewer_notes: str = ""
    reviewed_at: Optional[str] = None

    @property
    def is_reviewed(self) -> bool:
        return self.decision is not None

    @property
    def final_value(self) -> Any:
        if self.decision == "accept":
            return self.extracted_value
        elif self.decision == "correct":
            return self.corrected_value
        elif self.decision == "na":
            return None
        return self.extracted_value


# Canonical field order matching ground truth format
CANONICAL_FIELD_ORDER = [
    "fund_type",
    # Share classes (dynamic - inserted in order)
    "incentive_fee.has_incentive_fee",
    "incentive_fee.incentive_fee_pct",
    "incentive_fee.hurdle_rate_pct",
    "incentive_fee.hurdle_rate_as_stated",
    "incentive_fee.hurdle_rate_frequency",
    "incentive_fee.high_water_mark",
    "incentive_fee.has_catch_up",
    "incentive_fee.catch_up_rate_pct",
    "incentive_fee.catch_up_ceiling_pct",
    "incentive_fee.fee_basis",
    "incentive_fee.crystallization_frequency",
    "incentive_fee.underlying_fund_incentive_range",
    "expense_cap.has_expense_cap",
    "expense_cap.expense_cap_pct",
    "repurchase_terms.repurchase_frequency",
    "repurchase_terms.repurchase_amount_pct",
    "repurchase_terms.repurchase_basis",
    "repurchase_terms.repurchase_percentage_min",
    "repurchase_terms.repurchase_percentage_max",
    "repurchase_terms.lock_up_period_years",
    "repurchase_terms.early_repurchase_fee_pct",
    "leverage_limits.uses_leverage",
    "leverage_limits.max_leverage_pct",
    "leverage_limits.leverage_basis",
    "distribution_terms.distribution_frequency",
    "distribution_terms.default_distribution_policy",
    "allocation_targets.secondary_funds_min_pct",
    "allocation_targets.secondary_funds_max_pct",
    "allocation_targets.direct_investments_min_pct",
    "allocation_targets.direct_investments_max_pct",
    "allocation_targets.secondary_investments_min_pct",
    "concentration_limits.max_single_asset_pct",
    "concentration_limits.max_single_fund_pct",
    "concentration_limits.max_single_sector_pct",
]

# Field categories mapping
FIELD_CATEGORIES = {
    "fund_type": "Fund Metadata",
    "incentive_fee.has_incentive_fee": "Incentive Fee",
    "incentive_fee.incentive_fee_pct": "Incentive Fee",
    "incentive_fee.hurdle_rate_pct": "Incentive Fee",
    "incentive_fee.hurdle_rate_as_stated": "Incentive Fee",
    "incentive_fee.hurdle_rate_frequency": "Incentive Fee",
    "incentive_fee.high_water_mark": "Incentive Fee",
    "incentive_fee.has_catch_up": "Incentive Fee",
    "incentive_fee.catch_up_rate_pct": "Incentive Fee",
    "incentive_fee.catch_up_ceiling_pct": "Incentive Fee",
    "incentive_fee.fee_basis": "Incentive Fee",
    "incentive_fee.crystallization_frequency": "Incentive Fee",
    "incentive_fee.underlying_fund_incentive_range": "Incentive Fee",
    "expense_cap.has_expense_cap": "Expense Cap",
    "expense_cap.expense_cap_pct": "Expense Cap",
    "repurchase_terms.repurchase_frequency": "Repurchase Terms",
    "repurchase_terms.repurchase_amount_pct": "Repurchase Terms",
    "repurchase_terms.repurchase_basis": "Repurchase Terms",
    "repurchase_terms.repurchase_percentage_min": "Repurchase Terms",
    "repurchase_terms.repurchase_percentage_max": "Repurchase Terms",
    "repurchase_terms.lock_up_period_years": "Repurchase Terms",
    "repurchase_terms.early_repurchase_fee_pct": "Repurchase Terms",
    "leverage_limits.uses_leverage": "Leverage",
    "leverage_limits.max_leverage_pct": "Leverage",
    "leverage_limits.leverage_basis": "Leverage",
    "distribution_terms.distribution_frequency": "Distribution",
    "distribution_terms.default_distribution_policy": "Distribution",
    "allocation_targets.secondary_funds_min_pct": "Allocation Targets",
    "allocation_targets.secondary_funds_max_pct": "Allocation Targets",
    "allocation_targets.direct_investments_min_pct": "Allocation Targets",
    "allocation_targets.direct_investments_max_pct": "Allocation Targets",
    "allocation_targets.secondary_investments_min_pct": "Allocation Targets",
    "concentration_limits.max_single_asset_pct": "Concentration Limits",
    "concentration_limits.max_single_fund_pct": "Concentration Limits",
    "concentration_limits.max_single_sector_pct": "Concentration Limits",
}

# Human-readable display names
FIELD_DISPLAY_NAMES = {
    "fund_type": "Fund Type",
    "incentive_fee.has_incentive_fee": "Incentive Fee - Has Incentive Fee",
    "incentive_fee.incentive_fee_pct": "Incentive Fee - Fee Rate (%)",
    "incentive_fee.hurdle_rate_pct": "Incentive Fee - Hurdle Rate (%, annualized)",
    "incentive_fee.hurdle_rate_as_stated": "Incentive Fee - Hurdle Rate (as stated)",
    "incentive_fee.hurdle_rate_frequency": "Incentive Fee - Hurdle Rate Frequency",
    "incentive_fee.high_water_mark": "Incentive Fee - High Water Mark",
    "incentive_fee.has_catch_up": "Incentive Fee - Has Catch-Up",
    "incentive_fee.catch_up_rate_pct": "Incentive Fee - Catch-Up Rate (%)",
    "incentive_fee.catch_up_ceiling_pct": "Incentive Fee - Catch-Up Ceiling (%)",
    "incentive_fee.fee_basis": "Incentive Fee - Fee Basis",
    "incentive_fee.crystallization_frequency": "Incentive Fee - Crystallization Frequency",
    "incentive_fee.underlying_fund_incentive_range": "Incentive Fee - Underlying Fund Incentive Range",
    "expense_cap.has_expense_cap": "Expense Cap - Has Expense Cap",
    "expense_cap.expense_cap_pct": "Expense Cap - Cap Rate (%)",
    "repurchase_terms.repurchase_frequency": "Repurchase Terms - Frequency",
    "repurchase_terms.repurchase_amount_pct": "Repurchase Terms - Amount (%)",
    "repurchase_terms.repurchase_basis": "Repurchase Terms - Basis",
    "repurchase_terms.repurchase_percentage_min": "Repurchase Terms - Minimum (%)",
    "repurchase_terms.repurchase_percentage_max": "Repurchase Terms - Maximum (%)",
    "repurchase_terms.lock_up_period_years": "Repurchase Terms - Lock-Up Period (years)",
    "repurchase_terms.early_repurchase_fee_pct": "Repurchase Terms - Early Repurchase Fee (%)",
    "leverage_limits.uses_leverage": "Leverage - Uses Leverage",
    "leverage_limits.max_leverage_pct": "Leverage - Maximum (%)",
    "leverage_limits.leverage_basis": "Leverage - Basis",
    "distribution_terms.distribution_frequency": "Distribution - Frequency",
    "distribution_terms.default_distribution_policy": "Distribution - Default Policy",
    "allocation_targets.secondary_funds_min_pct": "Allocation Targets - Secondary Funds Min (%)",
    "allocation_targets.secondary_funds_max_pct": "Allocation Targets - Secondary Funds Max (%)",
    "allocation_targets.direct_investments_min_pct": "Allocation Targets - Direct Investments Min (%)",
    "allocation_targets.direct_investments_max_pct": "Allocation Targets - Direct Investments Max (%)",
    "allocation_targets.secondary_investments_min_pct": "Allocation Targets - Secondary Investments Min (%)",
    "concentration_limits.max_single_asset_pct": "Concentration Limits - Max Single Asset (%)",
    "concentration_limits.max_single_fund_pct": "Concentration Limits - Max Single Fund (%)",
    "concentration_limits.max_single_sector_pct": "Concentration Limits - Max Single Sector (%)",
}


def get_display_name(field_key: str) -> str:
    """Get human-readable display name for a field key."""
    if field_key in FIELD_DISPLAY_NAMES:
        return FIELD_DISPLAY_NAMES[field_key]
    # Handle share class fields: share_classes.Class I.minimum_initial_investment
    parts = field_key.split(".")
    if len(parts) == 3 and parts[0] == "share_classes":
        class_name = parts[1]
        field_name = parts[2].replace("_", " ").replace("pct", "(%)").title()
        return f"Share Classes - {class_name} - {field_name}"
    # Generic fallback
    return field_key.replace("_", " ").replace(".", " - ").title()


def get_field_sort_key(field_key: str) -> tuple[int, str]:
    """Return a sort key that follows canonical GT field order.

    Order: fund_type (0), share_classes (1), then remaining fields (2+).
    """
    if field_key == "fund_type":
        return (0, field_key)
    if field_key.startswith("share_classes."):
        return (1, field_key)
    try:
        idx = CANONICAL_FIELD_ORDER.index(field_key)
        return (idx + 2, field_key)
    except ValueError:
        return (999, field_key)


def _get_category(field_key: str) -> str:
    """Get category for a field key, handling share class fields."""
    if field_key in FIELD_CATEGORIES:
        return FIELD_CATEGORIES[field_key]
    if field_key.startswith("share_classes."):
        return "Share Classes"
    return "Other"


@dataclass
class ReviewSession:
    """Persistent review session."""
    fund_name: str
    experiment_dir: str
    cik: str = ""
    filing_date: str = ""
    filing_path: str = ""
    fields: dict[str, FieldReview] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_saved_at: Optional[str] = None

    @property
    def total_fields(self) -> int:
        return len(self.fields)

    @property
    def reviewed_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.is_reviewed)

    @property
    def accepted_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.decision == "accept")

    @property
    def corrected_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.decision == "correct")

    @property
    def na_count(self) -> int:
        return sum(1 for f in self.fields.values() if f.decision == "na")

    @property
    def categories(self) -> list[str]:
        cats = sorted(set(f.category for f in self.fields.values()))
        return cats

    def fields_by_category(self, category: str) -> list[FieldReview]:
        return [f for f in self.fields.values() if f.category == category]

    def save(self, path: Optional[Path] = None) -> Path:
        """Save session to JSON."""
        if path is None:
            path = Path(self.experiment_dir) / f"review_session_{_safe_name(self.fund_name)}.json"
        self.last_saved_at = datetime.now().isoformat()
        data = {
            "fund_name": self.fund_name,
            "experiment_dir": self.experiment_dir,
            "cik": self.cik,
            "filing_date": self.filing_date,
            "filing_path": self.filing_path,
            "created_at": self.created_at,
            "last_saved_at": self.last_saved_at,
            "fields": {k: asdict(v) for k, v in self.fields.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: Path) -> "ReviewSession":
        """Load session from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = cls(
            fund_name=data["fund_name"],
            experiment_dir=data["experiment_dir"],
            cik=data.get("cik", ""),
            filing_date=data.get("filing_date", ""),
            filing_path=data.get("filing_path", ""),
            created_at=data.get("created_at", ""),
            last_saved_at=data.get("last_saved_at"),
        )
        for key, fdata in data.get("fields", {}).items():
            session.fields[key] = FieldReview(**fdata)
        return session


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")[:50]


def build_session_from_results(
    experiment_dir: str,
    fund_name: str,
) -> ReviewSession:
    """
    Build a ReviewSession by parsing experiment results.json and trace.

    Extracts every field with its value, evidence, confidence, etc.
    """
    exp_path = Path(experiment_dir)
    results_path = exp_path / "results.json"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find the fund
    fund_results = data.get("fund_results", {})
    if fund_name not in fund_results:
        available = list(fund_results.keys())
        raise ValueError(f"Fund '{fund_name}' not found. Available: {available}")

    fund_data = fund_results[fund_name]
    extraction = fund_data.get("extraction", {})
    trace = fund_data.get("trace", {})
    field_decisions = trace.get("field_decisions", {})

    session = ReviewSession(
        fund_name=fund_name,
        experiment_dir=experiment_dir,
        cik=extraction.get("cik", fund_data.get("cik", "")),
        filing_date=extraction.get("filing_date", ""),
        filing_path=fund_data.get("filing_path", ""),
    )

    # Process field_decisions from trace (these have the dotted field names)
    for field_key, decision_info in field_decisions.items():
        category = _get_category(field_key)
        final_value = decision_info.get("final_value")
        source = decision_info.get("source", "")

        # Get evidence from extraction data
        evidence = _get_evidence(extraction, field_key, final_value)

        # Determine confidence from trace
        confidence = None
        if decision_info.get("llm_judge_validated") is True:
            confidence = "validated"
        elif decision_info.get("llm_judge_validated") is False:
            confidence = "rejected"

        grounded = decision_info.get("llm_judge_validated")

        session.fields[field_key] = FieldReview(
            field_key=field_key,
            category=category,
            extracted_value=final_value,
            evidence=evidence,
            confidence=confidence,
            grounded=grounded,
            tier=source,
            reasoning=None,
        )

    # If no field_decisions in trace, fall back to parsing extraction dict directly
    if not field_decisions:
        _build_fields_from_extraction(session, extraction)

    return session


def _get_evidence(extraction: dict, field_key: str, extracted_value: Any = None) -> Optional[str]:
    """Extract evidence string for a field from the extraction data.

    If the full evidence is long, tries to find a focused snippet around
    where the extracted value actually appears in the text.
    """
    # Special case: fund_type has no _evidence but has fund_type_flags from XBRL
    if field_key == "fund_type":
        flags = extraction.get("fund_type_flags", {})
        if flags:
            lines = [f"  {k}: {v}" for k, v in flags.items()]
            return "XBRL Fund Type Flags:\n" + "\n".join(lines)
        return None

    parts = field_key.split(".")

    # Share class fields: share_classes.ClassName.field_name
    # Evidence is shared under _evidence.share_classes
    if len(parts) == 3 and parts[0] == "share_classes":
        section_data = extraction.get("share_classes", {})
        if isinstance(section_data, dict):
            evidence_dict = section_data.get("_evidence", {})
            if isinstance(evidence_dict, dict):
                ev = evidence_dict.get("share_classes")
                if ev:
                    # Focus on the class name and value
                    class_name = parts[1]
                    return _focus_evidence(str(ev), class_name)
        return None

    if len(parts) >= 2:
        section = parts[0]
        field_name = parts[1]
        section_data = extraction.get(section, {})
        if isinstance(section_data, dict):
            evidence_dict = section_data.get("_evidence", {})
            if isinstance(evidence_dict, dict):
                ev = evidence_dict.get(field_name)
                if ev:
                    full_text = str(ev)
                    return _focus_evidence(full_text, extracted_value)
    return None


def _focus_evidence(text: str, value: Any, context_chars: int = 400) -> str:
    """If text is long, find the snippet around the extracted value."""
    if len(text) <= 800:
        return text

    if value is not None:
        search_str = str(value).strip()
        # Try exact match first
        idx = text.find(search_str)
        # Try with % suffix for percentages
        if idx == -1:
            idx = text.find(f"{search_str}%")
        # Try without trailing zeros (e.g. "5.0" -> "5")
        if idx == -1 and "." in search_str:
            idx = text.find(search_str.rstrip("0").rstrip("."))

        if idx != -1:
            start = max(0, idx - context_chars)
            end = min(len(text), idx + len(search_str) + context_chars)
            snippet = text[start:end]
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            return f"{prefix}{snippet}{suffix}"

    # Fallback: return first 800 chars
    return text[:800] + ("..." if len(text) > 800 else "")


def _build_fields_from_extraction(session: ReviewSession, extraction: dict) -> None:
    """Fallback: build fields from raw extraction dict when no trace is available."""
    # Non-field top-level keys to skip
    skip_keys = {
        "filing_id", "cik", "fund_name", "fund_type_flags",
        "xbrl_fees", "ensemble_metadata",
    }

    for section_key, section_data in extraction.items():
        if section_key in skip_keys:
            continue
        if not isinstance(section_data, dict):
            # Simple top-level field like fund_type
            field_key = section_key
            session.fields[field_key] = FieldReview(
                field_key=field_key,
                category=_get_category(field_key),
                extracted_value=section_data,
            )
            continue

        evidence_dict = section_data.get("_evidence", {})

        for field_name, value in section_data.items():
            if field_name.startswith("_"):
                continue
            field_key = f"{section_key}.{field_name}"
            evidence = None
            if isinstance(evidence_dict, dict):
                ev = evidence_dict.get(field_name)
                if ev:
                    evidence = _focus_evidence(str(ev), value)

            # Handle share_classes specially - flatten
            if section_key == "share_classes" and field_name == "share_classes" and isinstance(value, list):
                for sc in value:
                    if not isinstance(sc, dict):
                        continue
                    class_name = sc.get("class_name", "Unknown")
                    for sc_field, sc_val in sc.items():
                        if sc_field == "class_name":
                            continue
                        sc_key = f"share_classes.{class_name}.{sc_field}"
                        session.fields[sc_key] = FieldReview(
                            field_key=sc_key,
                            category="Share Classes",
                            extracted_value=sc_val,
                        )
                continue

            session.fields[field_key] = FieldReview(
                field_key=field_key,
                category=_get_category(field_key),
                extracted_value=value,
            )
