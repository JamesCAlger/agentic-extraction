"""
Fund Type Indexer - Build and maintain an index of interval/tender offer funds.

This module:
1. Queries SEC EDGAR for all N-2 filings
2. Downloads just enough of each filing to extract fund type flags
3. Maintains an incremental index that only processes new filings

Usage:
    # Initial build (one-time, ~2-4 hours for 13,000+ filings)
    python -m pipeline.index.fund_type_indexer --full-build

    # Incremental update (daily, ~1-5 minutes)
    python -m pipeline.index.fund_type_indexer --update
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..parse.ixbrl_parser import IXBRLParser, FundType

logger = logging.getLogger(__name__)

# SEC EDGAR configuration
SEC_BASE_URL = "https://www.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_FULL_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index"
SEC_USER_AGENT = "EvergreenPlatform contact@example.com"  # Required by SEC
SEC_RATE_LIMIT_DELAY = 0.1  # 100ms = 10 requests/second

# How much of each N-2 to download (fund type flags are on cover page)
MAX_DOWNLOAD_BYTES = 150_000  # 150KB should be plenty for cover page


@dataclass
class FundIndexEntry:
    """Entry for a single fund in the index."""
    cik: str
    name: str
    fund_type: str
    latest_filing_date: str
    accession_number: str
    flags: dict
    last_checked: str


@dataclass
class FundTypeIndex:
    """Complete index of fund types."""
    last_updated: str
    total_filings_processed: int
    funds: dict[str, FundIndexEntry]

    # Quick lookup lists (CIKs only)
    interval_funds: list[str]
    tender_offer_funds: list[str]
    bdcs: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "last_updated": self.last_updated,
            "total_filings_processed": self.total_filings_processed,
            "funds": {cik: asdict(entry) for cik, entry in self.funds.items()},
            "interval_funds": self.interval_funds,
            "tender_offer_funds": self.tender_offer_funds,
            "bdcs": self.bdcs,
            "summary": {
                "total_funds": len(self.funds),
                "interval_count": len(self.interval_funds),
                "tender_offer_count": len(self.tender_offer_funds),
                "bdc_count": len(self.bdcs),
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FundTypeIndex":
        """Load from dictionary."""
        funds = {}
        for cik, entry_data in data.get("funds", {}).items():
            funds[cik] = FundIndexEntry(**entry_data)

        return cls(
            last_updated=data.get("last_updated", ""),
            total_filings_processed=data.get("total_filings_processed", 0),
            funds=funds,
            interval_funds=data.get("interval_funds", []),
            tender_offer_funds=data.get("tender_offer_funds", []),
            bdcs=data.get("bdcs", []),
        )


class SECEdgarClient:
    """Client for SEC EDGAR API with rate limiting."""

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce SEC rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with rate limiting."""
        self._rate_limit()
        return self.session.get(url, **kwargs)

    def get_n2_filings_since(
        self,
        since_date: Optional[str] = None,
        max_results: int = 10000,
    ) -> list[dict]:
        """
        Get list of N-2 filings since a given date.

        Uses SEC EDGAR quarterly index files for reliable results.

        Args:
            since_date: ISO date string (YYYY-MM-DD), or None for all filings
            max_results: Maximum number of results to return

        Returns:
            List of filing metadata dicts with keys: cik, company_name,
            accession_number, filing_date, form_type
        """
        filings = []

        # Default start date if not specified
        start_date = since_date or "2019-01-01"
        start_year = int(start_date[:4])
        current_year = datetime.now(timezone.utc).year

        # Determine which quarters to fetch
        quarters_to_fetch = []
        for year in range(start_year, current_year + 1):
            for qtr in range(1, 5):
                quarters_to_fetch.append((year, qtr))

        logger.info(f"Fetching N-2 filings from {len(quarters_to_fetch)} quarters...")

        for year, qtr in quarters_to_fetch:
            if len(filings) >= max_results:
                break

            # Fetch quarterly index
            index_url = f"{SEC_FULL_INDEX_URL}/{year}/QTR{qtr}/company.idx"

            try:
                response = self.get(index_url)
                if response.status_code != 200:
                    logger.debug(f"No index for {year} QTR{qtr}")
                    continue

                # Parse index file
                # Format: Company Name | Form Type | CIK | Date Filed | Filename
                lines = response.text.split("\n")

                for line in lines:
                    # Skip header lines
                    if not line.strip() or line.startswith("-"):
                        continue

                    # Check for N-2 forms (including N-2/A, N-2 POS, etc.)
                    # The form type is in columns ~62-74 approximately
                    if " N-2 " not in line and " N-2/A " not in line:
                        continue

                    # Parse fixed-width format
                    # Company name: cols 0-61
                    # Form type: cols 62-73
                    # CIK: cols 74-85
                    # Date: cols 86-97
                    # Filename: cols 98+
                    try:
                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        # Find form type position (N-2 or N-2/A)
                        form_idx = None
                        for i, part in enumerate(parts):
                            if part in ["N-2", "N-2/A", "N-2ASR", "N-2MEF"]:
                                form_idx = i
                                break

                        if form_idx is None:
                            continue

                        # CIK is after form type
                        cik = parts[form_idx + 1].zfill(10)
                        filing_date = parts[form_idx + 2]
                        form_type = parts[form_idx]

                        # Company name is everything before form type
                        company_name = " ".join(parts[:form_idx]).strip()

                        # Filename/accession is after date
                        filename = parts[form_idx + 3] if len(parts) > form_idx + 3 else ""
                        # Extract accession number from filename path
                        acc_match = re.search(r"(\d{10}-\d{2}-\d{6})", filename)
                        accession = acc_match.group(1) if acc_match else ""

                        # Filter by date if specified
                        if since_date and filing_date < since_date:
                            continue

                        if cik and accession:
                            filings.append({
                                "cik": cik,
                                "company_name": company_name,
                                "accession_number": accession,
                                "filing_date": filing_date,
                                "form_type": form_type,
                            })

                    except (IndexError, ValueError) as e:
                        continue

            except Exception as e:
                logger.warning(f"Error fetching {year} QTR{qtr} index: {e}")
                continue

        # Deduplicate by (cik, accession_number)
        seen = set()
        unique_filings = []
        for f in filings:
            key = (f["cik"], f["accession_number"])
            if key not in seen:
                seen.add(key)
                unique_filings.append(f)

        logger.info(f"Found {len(unique_filings)} unique N-2 filings")
        return unique_filings[:max_results]

    def get_n2_filing_document_url(self, cik: str, accession_number: str) -> Optional[str]:
        """
        Get the URL of the primary N-2 document.

        Args:
            cik: Fund CIK (10 digits)
            accession_number: Filing accession number

        Returns:
            URL of primary document, or None if not found
        """
        # Format accession number for URL (remove dashes)
        acc_nodash = accession_number.replace("-", "")

        # Normalize CIK (remove leading zeros for URL)
        cik_normalized = str(int(cik))

        # Get filing index page
        index_url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik_normalized}/{acc_nodash}/{accession_number}-index.htm"

        response = self.get(index_url)
        if response.status_code != 200:
            # Try with full CIK
            index_url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik}/{acc_nodash}/{accession_number}-index.htm"
            response = self.get(index_url)
            if response.status_code != 200:
                return None

        # Parse to find primary document
        soup = BeautifulSoup(response.content, "html.parser")

        # Look for N-2 document in table rows
        # SEC filing index has a table with: Seq, Description, Document, Type, Size
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                doc_type = cells[3].get_text().strip().upper() if len(cells) > 3 else ""

                # Look for N-2 type (not exhibits like EX-99, not EX-FILING FEES)
                if doc_type in ["N-2", "N-2/A", "N-2 POS", "N-2 MEF"]:
                    link = row.find("a")
                    if link:
                        href = link.get("href", "")
                        # Handle iXBRL wrapper URLs like /ix?doc=/Archives/...
                        if "/ix?doc=" in href:
                            # Extract the actual document path
                            doc_path = href.split("/ix?doc=")[-1]
                            return f"{SEC_BASE_URL}{doc_path}"
                        elif href.endswith((".htm", ".html")):
                            if href.startswith("http"):
                                return href
                            elif href.startswith("/"):
                                return f"{SEC_BASE_URL}{href}"
                            else:
                                return urljoin(index_url, href)

        # Fallback: look for any primary-looking document link
        for link in soup.find_all("a"):
            href = link.get("href", "")

            # Skip non-document links
            if not href.endswith((".htm", ".html")):
                continue
            if "index" in href.lower():
                continue
            if any(skip in href.lower() for skip in ["ex-", "exhibit", "graphic"]):
                continue

            # Handle iXBRL wrapper
            if "/ix?doc=" in href:
                doc_path = href.split("/ix?doc=")[-1]
                return f"{SEC_BASE_URL}{doc_path}"

            if href.startswith("http"):
                return href
            elif href.startswith("/"):
                return f"{SEC_BASE_URL}{href}"
            else:
                return urljoin(index_url, href)

        return None

    def download_document_head(
        self,
        url: str,
        max_bytes: int = MAX_DOWNLOAD_BYTES
    ) -> Optional[str]:
        """
        Download just the first N bytes of a document.

        Args:
            url: Document URL
            max_bytes: Maximum bytes to download

        Returns:
            Document content (partial), or None on error
        """
        try:
            self._rate_limit()
            response = self.session.get(
                url,
                headers={"Range": f"bytes=0-{max_bytes}"},
                timeout=30,
            )

            if response.status_code in (200, 206):  # 206 = Partial Content
                return response.text
            else:
                logger.warning(f"Failed to download {url}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None


class FundTypeIndexer:
    """
    Builds and maintains an index of fund types from N-2 filings.
    """

    def __init__(
        self,
        index_path: str = "data/fund_type_index.json",
        cache_dir: str = "data/cache/n2_headers",
    ):
        self.index_path = Path(index_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = SECEdgarClient()
        self.parser = IXBRLParser()

        # Load existing index or create new
        self.index = self._load_index()

    def _load_index(self) -> FundTypeIndex:
        """Load existing index or create empty one."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                data = json.load(f)
            return FundTypeIndex.from_dict(data)

        return FundTypeIndex(
            last_updated="",
            total_filings_processed=0,
            funds={},
            interval_funds=[],
            tender_offer_funds=[],
            bdcs=[],
        )

    def _save_index(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump(self.index.to_dict(), f, indent=2)

    def _get_cached_document(self, cik: str, accession: str) -> Optional[str]:
        """Get cached document header if available."""
        cache_file = self.cache_dir / f"{cik}_{accession}.html"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def _cache_document(self, cik: str, accession: str, content: str):
        """Cache document header."""
        cache_file = self.cache_dir / f"{cik}_{accession}.html"
        cache_file.write_text(content, encoding="utf-8")

    def process_filing(self, filing: dict) -> Optional[FundIndexEntry]:
        """
        Process a single N-2 filing and extract fund type.

        Args:
            filing: Filing metadata dict

        Returns:
            FundIndexEntry or None if processing failed
        """
        cik = filing["cik"]
        accession = filing["accession_number"]

        # Check cache first
        html_content = self._get_cached_document(cik, accession)

        if not html_content:
            # Get document URL
            doc_url = self.client.get_n2_filing_document_url(cik, accession)
            if not doc_url:
                logger.warning(f"Could not find document for {cik}/{accession}")
                return None

            # Download document header
            html_content = self.client.download_document_head(doc_url)
            if not html_content:
                return None

            # Cache for future use
            self._cache_document(cik, accession, html_content)

        # Extract fund type flags
        try:
            fund_type, flags = self.parser.extract_fund_type(html_content)
        except Exception as e:
            logger.error(f"Error parsing {cik}/{accession}: {e}")
            return None

        return FundIndexEntry(
            cik=cik,
            name=filing.get("company_name", ""),
            fund_type=fund_type.value,
            latest_filing_date=filing.get("filing_date", ""),
            accession_number=accession,
            flags=flags,
            last_checked=datetime.now(timezone.utc).isoformat(),
        )

    def build_full_index(
        self,
        max_filings: int = 20000,
        workers: int = 5,
        progress_callback=None,
    ) -> dict:
        """
        Build complete index from all N-2 filings.

        Args:
            max_filings: Maximum number of filings to process
            workers: Number of parallel download workers
            progress_callback: Optional callback(processed, total, current_filing)

        Returns:
            Summary statistics
        """
        logger.info("Fetching list of all N-2 filings...")
        filings = self.client.get_n2_filings_since(since_date=None, max_results=max_filings)
        logger.info(f"Found {len(filings)} N-2 filings")

        # Group by CIK and keep only latest filing per fund
        latest_by_cik = {}
        for filing in filings:
            cik = filing["cik"]
            if cik not in latest_by_cik:
                latest_by_cik[cik] = filing
            elif filing["filing_date"] > latest_by_cik[cik]["filing_date"]:
                latest_by_cik[cik] = filing

        unique_filings = list(latest_by_cik.values())
        logger.info(f"Processing {len(unique_filings)} unique funds (latest filing each)")

        # Process filings
        processed = 0
        errors = 0

        for filing in unique_filings:
            processed += 1

            if progress_callback:
                progress_callback(processed, len(unique_filings), filing)

            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(unique_filings)} filings...")
                self._save_index()  # Checkpoint

            entry = self.process_filing(filing)
            if entry:
                self.index.funds[entry.cik] = entry
            else:
                errors += 1

        # Rebuild quick lookup lists
        self._rebuild_lookup_lists()

        # Update metadata
        self.index.total_filings_processed = len(filings)
        self.index.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Save final index
        self._save_index()

        return {
            "total_filings_found": len(filings),
            "unique_funds_processed": len(unique_filings),
            "errors": errors,
            "interval_funds": len(self.index.interval_funds),
            "tender_offer_funds": len(self.index.tender_offer_funds),
            "bdcs": len(self.index.bdcs),
        }

    def update_index(self, days_back: int = 7) -> dict:
        """
        Incrementally update index with recent filings.

        Args:
            days_back: How many days back to check for new filings

        Returns:
            Summary statistics
        """
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        logger.info(f"Fetching N-2 filings since {since_date}...")
        filings = self.client.get_n2_filings_since(since_date=since_date)
        logger.info(f"Found {len(filings)} recent N-2 filings")

        new_funds = 0
        updated_funds = 0
        errors = 0

        for filing in filings:
            cik = filing["cik"]

            # Check if we already have a newer filing for this fund
            existing = self.index.funds.get(cik)
            if existing and existing.latest_filing_date >= filing["filing_date"]:
                continue  # Skip older filing

            entry = self.process_filing(filing)
            if entry:
                if cik in self.index.funds:
                    updated_funds += 1
                else:
                    new_funds += 1
                self.index.funds[cik] = entry
            else:
                errors += 1

        # Rebuild lookup lists
        self._rebuild_lookup_lists()

        # Update metadata
        self.index.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Save
        self._save_index()

        return {
            "filings_checked": len(filings),
            "new_funds": new_funds,
            "updated_funds": updated_funds,
            "errors": errors,
            "total_interval_funds": len(self.index.interval_funds),
            "total_tender_offer_funds": len(self.index.tender_offer_funds),
        }

    def _rebuild_lookup_lists(self):
        """Rebuild quick lookup lists from funds dict."""
        self.index.interval_funds = [
            cik for cik, entry in self.index.funds.items()
            if entry.fund_type == FundType.INTERVAL_FUND.value
        ]
        self.index.tender_offer_funds = [
            cik for cik, entry in self.index.funds.items()
            if entry.fund_type == FundType.TENDER_OFFER_FUND.value
        ]
        self.index.bdcs = [
            cik for cik, entry in self.index.funds.items()
            if entry.fund_type == FundType.BDC.value
        ]

    def get_evergreen_ciks(self) -> list[str]:
        """Get list of all interval and tender offer fund CIKs."""
        return self.index.interval_funds + self.index.tender_offer_funds

    def get_fund_info(self, cik: str) -> Optional[FundIndexEntry]:
        """Get fund info by CIK."""
        return self.index.funds.get(cik)


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Build fund type index from N-2 filings")
    parser.add_argument("--full-build", action="store_true", help="Build complete index from scratch")
    parser.add_argument("--update", action="store_true", help="Incremental update with recent filings")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back for updates")
    parser.add_argument("--max-filings", type=int, default=20000, help="Max filings to process")
    parser.add_argument("--output", default="data/fund_type_index.json", help="Output index path")

    args = parser.parse_args()

    indexer = FundTypeIndexer(index_path=args.output)

    if args.full_build:
        print("Building full index (this may take 2-4 hours)...")
        result = indexer.build_full_index(max_filings=args.max_filings)
        print(f"\nResults:")
        print(f"  Total filings found: {result['total_filings_found']}")
        print(f"  Unique funds processed: {result['unique_funds_processed']}")
        print(f"  Interval funds: {result['interval_funds']}")
        print(f"  Tender offer funds: {result['tender_offer_funds']}")
        print(f"  BDCs: {result['bdcs']}")
        print(f"  Errors: {result['errors']}")

    elif args.update:
        print(f"Updating index (checking last {args.days_back} days)...")
        result = indexer.update_index(days_back=args.days_back)
        print(f"\nResults:")
        print(f"  Filings checked: {result['filings_checked']}")
        print(f"  New funds: {result['new_funds']}")
        print(f"  Updated funds: {result['updated_funds']}")
        print(f"  Total interval funds: {result['total_interval_funds']}")
        print(f"  Total tender offer funds: {result['total_tender_offer_funds']}")

    else:
        # Show current index stats
        print(f"Current index: {args.output}")
        print(f"  Last updated: {indexer.index.last_updated}")
        print(f"  Total funds: {len(indexer.index.funds)}")
        print(f"  Interval funds: {len(indexer.index.interval_funds)}")
        print(f"  Tender offer funds: {len(indexer.index.tender_offer_funds)}")
        print(f"  BDCs: {len(indexer.index.bdcs)}")

        evergreen = indexer.get_evergreen_ciks()
        print(f"\n  Total evergreen (interval + tender): {len(evergreen)}")


if __name__ == "__main__":
    main()
