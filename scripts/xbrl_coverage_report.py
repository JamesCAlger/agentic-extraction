"""
Generate XBRL coverage report showing what data is captured vs available.
"""

import sys
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.parse.ixbrl_parser import XBRLValueExtractor


def main():
    data_dir = Path("data/raw")
    extractor = XBRLValueExtractor()

    # Currently mapped
    mapped_numeric = set(extractor.TAG_FIELD_MAP.keys())
    mapped_text = set(extractor.TEXT_BLOCK_FIELDS.keys())

    # Scan all filings
    all_numeric = set()
    all_nonnumeric = set()

    for filing in data_dir.glob("**/primary.html"):
        with open(filing, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")

        for elem in soup.find_all("ix:nonfraction"):
            tag = elem.get("name", "")
            if tag:
                all_numeric.add(tag)

        for elem in soup.find_all("ix:nonnumeric"):
            tag = elem.get("name", "")
            if tag:
                all_nonnumeric.add(tag)

    # Calculate coverage
    numeric_mapped = mapped_numeric & all_numeric
    numeric_unmapped = all_numeric - mapped_numeric

    text_blocks_in_filings = {t for t in all_nonnumeric if "TextBlock" in t or "Text" in t}
    text_mapped = mapped_text & text_blocks_in_filings
    text_unmapped = text_blocks_in_filings - mapped_text

    print("=" * 70)
    print("XBRL COVERAGE REPORT")
    print("=" * 70)

    print(f"\n{'='*70}")
    print("NUMERIC FIELDS")
    print(f"{'='*70}")
    print(f"Total unique tags in filings: {len(all_numeric)}")
    print(f"Currently mapped: {len(numeric_mapped)} ({len(numeric_mapped)/len(all_numeric)*100:.0f}%)")
    print(f"Unmapped: {len(numeric_unmapped)}")

    print(f"\nMapped tags:")
    for tag in sorted(numeric_mapped):
        field = extractor.TAG_FIELD_MAP[tag]
        print(f"  {tag} -> {field}")

    print(f"\nUnmapped tags (not in parser):")
    for tag in sorted(numeric_unmapped):
        print(f"  {tag}")

    print(f"\n{'='*70}")
    print("TEXT BLOCKS")
    print(f"{'='*70}")
    print(f"Total text blocks in filings: {len(text_blocks_in_filings)}")
    print(f"Currently mapped: {len(text_mapped)} ({len(text_mapped)/len(text_blocks_in_filings)*100:.0f}%)")
    print(f"Unmapped: {len(text_unmapped)}")

    print(f"\nMapped text blocks:")
    for tag in sorted(text_mapped):
        field = extractor.TEXT_BLOCK_FIELDS[tag]
        print(f"  {tag} -> {field}")

    print(f"\nUnmapped text blocks:")
    for tag in sorted(text_unmapped):
        print(f"  {tag}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_available = len(all_numeric) + len(text_blocks_in_filings)
    total_mapped = len(numeric_mapped) + len(text_mapped)
    print(f"Total XBRL data points available: {total_available}")
    print(f"Total mapped by parser: {total_mapped} ({total_mapped/total_available*100:.0f}%)")


if __name__ == "__main__":
    main()
