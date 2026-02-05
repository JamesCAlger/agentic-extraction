"""
Compare combined vs per-section extraction modes.

Runs extraction on all 3 funds using both modes and compares:
1. Datapoints captured
2. Accuracy/completeness
3. LLM calls made
4. Estimated cost
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from pipeline.parse.processor import process_filing
from pipeline.extract.extractor import DocumentExtractor


# Corrected fund-to-filing mapping based on actual filing content
FUND_FILINGS = {
    "Hamilton Lane Private Assets": "data/raw/0001803491/2025-07-29_0001213900-25-068734",
    "Blackstone Private Credit": "data/raw/0002032432/2025-03-05_0001193125-25-047335",
    "StepStone Private Markets": "data/raw/0001789470/2025-08-19_0001193125-25-183488",
}

# Key datapoints to track
KEY_DATAPOINTS = [
    "fund_type",
    "share_classes.share_classes",
    "incentive_fee.incentive_fee_pct",
    "incentive_fee.high_water_mark",
    "incentive_fee.hurdle_rate_pct",
    "expense_cap.has_expense_cap",
    "repurchase_terms.repurchase_frequency",
    "repurchase_terms.repurchase_amount_pct",
    "repurchase_terms.lock_up_period_years",
    "repurchase_terms.early_repurchase_fee_pct",
    "leverage_limits.uses_leverage",
    "leverage_limits.max_leverage_pct",
    "distribution_terms.distribution_frequency",
]

# GPT-4o-mini pricing (per 1M tokens)
INPUT_PRICE_PER_1M = 0.15   # $0.15 per 1M input tokens
OUTPUT_PRICE_PER_1M = 0.60  # $0.60 per 1M output tokens


def get_nested_value(data: dict, path: str):
    """Get a nested value from a dict using dot notation."""
    parts = path.split(".")
    current = data
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def extract_fund(fund_name: str, raw_dir: Path, api_key: str, per_section: bool) -> dict:
    """Run extraction on a fund with specified mode."""
    mode_str = "per-section" if per_section else "combined"
    print(f"\n  [{mode_str}] Processing {fund_name}...")

    # Process document
    doc_map, chunked_doc, xbrl_values = process_filing(raw_dir)

    # Read HTML for fallback extraction
    html_path = raw_dir / "primary.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Run extraction
    extractor = DocumentExtractor(
        api_key=api_key,
        model="gpt-4o-mini",
        enable_grounding=True,
        enable_observability=True,
        per_section_extraction=per_section,
    )

    result = extractor.extract(
        chunked_doc=chunked_doc,
        xbrl_values=xbrl_values,
        fund_name=fund_name,
        html_content=html_content,
    )

    # Extract key datapoints
    datapoints = {}
    captured = 0
    for path in KEY_DATAPOINTS:
        value = get_nested_value(result, path)
        datapoints[path] = value
        if value is not None:
            captured += 1

    return {
        "fund_name": fund_name,
        "mode": mode_str,
        "datapoints": datapoints,
        "captured": captured,
        "total": len(KEY_DATAPOINTS),
        "chunks_processed": result.get("chunks_processed", 0),
        "grounding": result.get("grounding", {}),
    }


def estimate_cost(chunks_processed: int, avg_tokens_per_chunk: int = 500) -> float:
    """Estimate API cost based on chunks processed."""
    # Assume ~500 tokens input per chunk, ~200 tokens output per call
    input_tokens = chunks_processed * avg_tokens_per_chunk
    output_tokens = chunks_processed * 200  # Rough estimate

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M

    return input_cost + output_cost


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    print("=" * 70)
    print("EXTRACTION MODE COMPARISON")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {"combined": {}, "per_section": {}}

    for fund_name, raw_path in FUND_FILINGS.items():
        raw_dir = Path(raw_path)
        if not raw_dir.exists():
            print(f"\nSkipping {fund_name} - path not found")
            continue

        print(f"\n{'='*70}")
        print(f"FUND: {fund_name}")
        print(f"{'='*70}")

        try:
            # Run combined mode
            results["combined"][fund_name] = extract_fund(
                fund_name, raw_dir, api_key, per_section=False
            )

            # Run per-section mode
            results["per_section"][fund_name] = extract_fund(
                fund_name, raw_dir, api_key, per_section=True
            )

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Per-fund comparison
    for fund_name in results["combined"].keys():
        combined = results["combined"].get(fund_name, {})
        per_section = results["per_section"].get(fund_name, {})

        print(f"\n--- {fund_name} ---")
        print(f"  Combined mode:")
        print(f"    Captured: {combined.get('captured', 0)}/{combined.get('total', 0)}")
        print(f"    Chunks: {combined.get('chunks_processed', 0)}")
        print(f"    Est. cost: ${estimate_cost(combined.get('chunks_processed', 0)):.4f}")

        print(f"  Per-section mode:")
        print(f"    Captured: {per_section.get('captured', 0)}/{per_section.get('total', 0)}")
        print(f"    Chunks: {per_section.get('chunks_processed', 0)}")
        print(f"    Est. cost: ${estimate_cost(per_section.get('chunks_processed', 0)):.4f}")

        # Field-by-field comparison
        combined_dp = combined.get("datapoints", {})
        per_section_dp = per_section.get("datapoints", {})

        improved = []
        regressed = []
        for path in KEY_DATAPOINTS:
            c_val = combined_dp.get(path)
            p_val = per_section_dp.get(path)
            if c_val is None and p_val is not None:
                improved.append(path)
            elif c_val is not None and p_val is None:
                regressed.append(path)

        if improved:
            print(f"  Improvements: {improved}")
        if regressed:
            print(f"  Regressions: {regressed}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)

    total_combined_captured = sum(r.get("captured", 0) for r in results["combined"].values())
    total_combined_chunks = sum(r.get("chunks_processed", 0) for r in results["combined"].values())
    total_per_section_captured = sum(r.get("captured", 0) for r in results["per_section"].values())
    total_per_section_chunks = sum(r.get("chunks_processed", 0) for r in results["per_section"].values())
    total_possible = len(KEY_DATAPOINTS) * len(results["combined"])

    print(f"\nCombined mode:")
    print(f"  Total captured: {total_combined_captured}/{total_possible} ({100*total_combined_captured/total_possible:.1f}%)")
    print(f"  Total chunks: {total_combined_chunks}")
    print(f"  Estimated cost: ${estimate_cost(total_combined_chunks):.4f}")

    print(f"\nPer-section mode:")
    print(f"  Total captured: {total_per_section_captured}/{total_possible} ({100*total_per_section_captured/total_possible:.1f}%)")
    print(f"  Total chunks: {total_per_section_chunks}")
    print(f"  Estimated cost: ${estimate_cost(total_per_section_chunks):.4f}")

    if total_combined_chunks > 0:
        cost_increase = (total_per_section_chunks - total_combined_chunks) / total_combined_chunks * 100
        print(f"\nCost increase: {cost_increase:.1f}%")

    # Save results
    output_path = Path("data/extraction_mode_comparison.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "combined": {
            name: {
                "captured": data.get("captured"),
                "total": data.get("total"),
                "chunks_processed": data.get("chunks_processed"),
                "datapoints": data.get("datapoints"),
            }
            for name, data in results["combined"].items()
        },
        "per_section": {
            name: {
                "captured": data.get("captured"),
                "total": data.get("total"),
                "chunks_processed": data.get("chunks_processed"),
                "datapoints": data.get("datapoints"),
            }
            for name, data in results["per_section"].items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
