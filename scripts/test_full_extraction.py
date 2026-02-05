"""
Test full extraction pipeline on all 3 funds.

Runs Tiers 0-3 and captures all datapoints for comparison.
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
    "share_classes.share_classes",  # List of share classes with minimums
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


def extract_fund(fund_name: str, raw_dir: Path, api_key: str) -> dict:
    """Run full extraction on a fund and return results."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING: {fund_name}")
    print(f"{'='*70}")

    # Process document
    print("\nProcessing document...")
    doc_map, chunked_doc, xbrl_values = process_filing(raw_dir)
    print(f"  Sections: {len(chunked_doc.chunked_sections)}")
    print(f"  Chunks: {chunked_doc.total_chunks}")
    print(f"  Tokens: {chunked_doc.total_tokens:,}")

    # Read HTML for fallback extraction
    html_path = raw_dir / "primary.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Run extraction
    print("\nRunning tiered extraction...")
    extractor = DocumentExtractor(
        api_key=api_key,
        model="gpt-4o-mini",
        enable_grounding=True,
        enable_observability=True,
    )

    result = extractor.extract(
        chunked_doc=chunked_doc,
        xbrl_values=xbrl_values,
        fund_name=fund_name,
        html_content=html_content,
    )

    # Extract key datapoints
    print("\nKey datapoints extracted:")
    datapoints = {}
    for path in KEY_DATAPOINTS:
        value = get_nested_value(result, path)
        datapoints[path] = value
        status = "[OK]" if value is not None else "[--]"
        display_val = str(value)[:50] if value is not None else "null"
        print(f"  {status} {path}: {display_val}")

    return {
        "fund_name": fund_name,
        "extraction_result": result,
        "datapoints": datapoints,
        "stats": {
            "sections": len(chunked_doc.chunked_sections),
            "chunks": chunked_doc.total_chunks,
            "tokens": chunked_doc.total_tokens,
        },
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    print("=" * 70)
    print("FULL EXTRACTION TEST - ALL FUNDS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}
    for fund_name, raw_path in FUND_FILINGS.items():
        raw_dir = Path(raw_path)
        if not raw_dir.exists():
            print(f"\nSkipping {fund_name} - path not found")
            continue

        try:
            results[fund_name] = extract_fund(fund_name, raw_dir, api_key)
        except Exception as e:
            print(f"\nERROR extracting {fund_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    # Header
    fund_names = list(results.keys())
    print(f"\n{'Datapoint':<45} " + " ".join(f"{name[:15]:>17}" for name in fund_names))
    print("-" * (45 + 18 * len(fund_names)))

    # Count captured datapoints
    captured_counts = {name: 0 for name in fund_names}
    total_datapoints = len(KEY_DATAPOINTS)

    for path in KEY_DATAPOINTS:
        row = f"{path:<45} "
        for fund_name in fund_names:
            value = results[fund_name]["datapoints"].get(path)
            if value is not None:
                captured_counts[fund_name] += 1
                display = str(value)[:15] if not isinstance(value, list) else f"[{len(value)} items]"
            else:
                display = "null"
            row += f"{display:>17} "
        print(row)

    # Totals
    print("-" * (45 + 18 * len(fund_names)))
    row = f"{'CAPTURED':<45} "
    for fund_name in fund_names:
        count = captured_counts[fund_name]
        row += f"{count}/{total_datapoints}".rjust(17) + " "
    print(row)

    # Save results
    output_path = Path("data/extraction_comparison_500tok.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "chunk_size": 500,
        "results": {
            name: {
                "datapoints": data["datapoints"],
                "stats": data["stats"],
            }
            for name, data in results.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
