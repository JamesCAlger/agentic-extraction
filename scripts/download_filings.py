"""
Download SEC filings for specified funds.

Downloads N-2 (registration) or N-CSR (semi-annual) filings from SEC EDGAR.
Stores in format: data/raw/{CIK}/{filing_date}_{accession_number}/primary.html
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

# SEC requires a User-Agent header with contact info
USER_AGENT = "EvergreenFundsDataPlatform research@example.com"
SEC_BASE_URL = "https://www.sec.gov"
EDGAR_DATA_URL = "https://data.sec.gov"

# Rate limit: SEC allows 10 requests/second, we'll be conservative
REQUEST_DELAY = 0.15  # 150ms between requests

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def get_sec_headers():
    """Return headers required by SEC EDGAR."""
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


def get_company_filings(cik: str, filing_types: list[str] = None) -> list[dict]:
    """
    Get list of filings for a company from SEC EDGAR.

    Args:
        cik: Company CIK (will be zero-padded to 10 digits)
        filing_types: List of form types to filter (e.g., ["N-2", "N-CSR"])

    Returns:
        List of filing metadata dicts
    """
    if filing_types is None:
        filing_types = ["N-2", "N-CSR", "N-2/A", "N-CSR/A"]

    # Zero-pad CIK to 10 digits
    cik_padded = cik.lstrip("0").zfill(10)

    # SEC submissions endpoint
    url = f"{EDGAR_DATA_URL}/submissions/CIK{cik_padded}.json"

    headers = get_sec_headers()
    headers["Host"] = "data.sec.gov"

    print(f"Fetching filings list from: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    company_name = data.get("name", "Unknown")
    print(f"Company: {company_name}")

    # Extract recent filings
    recent = data.get("filings", {}).get("recent", {})

    filings = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in filing_types:
            filings.append({
                "form": form,
                "filing_date": dates[i],
                "accession_number": accessions[i],
                "primary_document": primary_docs[i],
                "company_name": company_name,
            })

    print(f"Found {len(filings)} filings of types {filing_types}")
    return filings


def download_filing(cik: str, filing: dict, output_dir: Path) -> Path:
    """
    Download a single filing's primary document.

    Args:
        cik: Company CIK
        filing: Filing metadata dict
        output_dir: Base output directory

    Returns:
        Path to downloaded file
    """
    cik_padded = cik.lstrip("0").zfill(10)
    accession_clean = filing["accession_number"].replace("-", "")

    # Construct URL to primary document
    primary_doc = filing["primary_document"]
    url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik_padded}/{accession_clean}/{primary_doc}"

    # Create output directory
    folder_name = f"{filing['filing_date']}_{filing['accession_number']}"
    filing_dir = output_dir / cik_padded / folder_name
    filing_dir.mkdir(parents=True, exist_ok=True)

    # Download file
    output_path = filing_dir / "primary.html"

    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    print(f"  Downloading: {url}")
    headers = get_sec_headers()
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Save file
    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"  Saved to: {output_path}")
    return output_path


def download_fund_filings(cik: str, max_filings: int = 2) -> list[Path]:
    """
    Download filings for a fund.

    Args:
        cik: Fund CIK
        max_filings: Maximum number of filings to download

    Returns:
        List of paths to downloaded files
    """
    print(f"\n{'='*60}")
    print(f"Processing CIK: {cik}")
    print(f"{'='*60}")

    # Get filing list
    filings = get_company_filings(cik)

    if not filings:
        print(f"No N-2/N-CSR filings found for CIK {cik}")
        return []

    # Download up to max_filings
    downloaded = []
    for i, filing in enumerate(filings[:max_filings]):
        print(f"\nFiling {i+1}/{min(len(filings), max_filings)}: {filing['form']} ({filing['filing_date']})")
        try:
            path = download_filing(cik, filing, DATA_DIR)
            downloaded.append(path)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  Error downloading: {e}")

    return downloaded


def main():
    """Download filings for recommended ground truth funds."""

    # Funds to download
    funds = [
        {
            "cik": "0002058263",
            "name": "Carlyle AlpInvest Private Markets Secondaries Fund",
            "strategy": "Private Equity (Secondaries)",
        },
        {
            "cik": "0001896329",
            "name": "PIMCO Flexible Real Estate Income Fund",
            "strategy": "Real Estate",
        },
        {
            "cik": "0002059436",
            "name": "Blue Owl Alternative Credit Fund",
            "strategy": "Private Credit",
        },
    ]

    print("SEC Filing Downloader")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")
    print(f"Funds to download: {len(funds)}")

    all_downloaded = []

    for fund in funds:
        time.sleep(REQUEST_DELAY)  # Rate limiting between funds
        downloaded = download_fund_filings(fund["cik"], max_filings=2)
        all_downloaded.extend(downloaded)

        print(f"\n{fund['name']}: Downloaded {len(downloaded)} filings")

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files downloaded: {len(all_downloaded)}")
    for path in all_downloaded:
        print(f"  - {path}")

    return all_downloaded


if __name__ == "__main__":
    main()
