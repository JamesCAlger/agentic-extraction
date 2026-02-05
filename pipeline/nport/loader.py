"""
N-PORT Bulk Data Loader

Downloads SEC N-PORT quarterly bulk datasets and filters to evergreen funds.

Usage:
    # Download and filter latest quarter
    python -m pipeline.nport.loader --latest

    # Download specific quarter
    python -m pipeline.nport.loader --quarter 2025q3

    # Download all available quarters
    python -m pipeline.nport.loader --all

Data source: https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets
"""

import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# SEC rate limit: 10 requests/second, we'll be conservative
REQUEST_DELAY = 0.15
SEC_USER_AGENT = "EvergreenFundsDataPlatform contact@example.com"

# Base URL for N-PORT bulk data
NPORT_DATA_URL = "https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets"
NPORT_ARCHIVE_BASE = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"

# Tables we care about for evergreen fund analysis
PRIORITY_TABLES = [
    "SUBMISSION",           # Filing metadata
    "REGISTRANT",           # Fund identity (CIK, name, LEI)
    "FUND_REPORTED_INFO",   # NAV, assets, flows, returns
    "MONTHLY_TOTAL_RETURN", # Monthly returns by class
    "FUND_REPORTED_HOLDING",# Individual holdings
    "IDENTIFIERS",          # ISIN, ticker for holdings
    "DEBT_SECURITY",        # Maturity, coupon, default status
]

# All available tables in N-PORT dataset
ALL_TABLES = [
    "SUBMISSION", "REGISTRANT", "FUND_REPORTED_INFO", "INTEREST_RATE_RISK",
    "BORROWER", "BORROW_AGGREGATE", "MONTHLY_TOTAL_RETURN",
    "MONTHLY_RETURN_CAT_INSTRUMENT", "FUND_VAR_INFO", "FUND_REPORTED_HOLDING",
    "IDENTIFIERS", "DEBT_SECURITY", "DEBT_SECURITY_REF_INSTRUMENT",
    "CONVERTIBLE_SECURITY_CURRENCY", "REPURCHASE_AGREEMENT",
    "REPURCHASE_COUNTERPARTY", "REPURCHASE_COLLATERAL",
    "DERIVATIVE_COUNTERPARTY", "SWAPTION_OPTION_WARNT_DERIV",
    "DESC_REF_INDEX_BASKET", "DESC_REF_INDEX_COMPONENT", "DESC_REF_OTHER",
    "FUT_FWD_NONFOREIGNCUR_CONTRACT", "FWD_FOREIGNCUR_CONTRACT_SWAP",
    "NONFOREIGN_EXCHANGE_SWAP", "FLOATING_RATE_RESET_TENOR", "OTHER_DERIV",
    "OTHER_DERIV_NOTIONAL_AMOUNT", "SECURITIES_LENDING", "EXPLANATORY_NOTE",
]


@dataclass
class QuarterData:
    """Container for a quarter's N-PORT data."""
    quarter: str  # e.g., "2025q3"
    submission: Optional[pd.DataFrame] = None
    registrant: Optional[pd.DataFrame] = None
    fund_reported_info: Optional[pd.DataFrame] = None
    monthly_total_return: Optional[pd.DataFrame] = None
    fund_reported_holding: Optional[pd.DataFrame] = None
    identifiers: Optional[pd.DataFrame] = None
    debt_security: Optional[pd.DataFrame] = None


class NPortLoader:
    """
    Downloads and processes SEC N-PORT bulk data for evergreen funds.

    The SEC provides pre-extracted N-PORT data in tab-delimited format,
    updated quarterly. This loader downloads the data and filters it
    to only include evergreen funds (interval and tender offer funds).
    """

    def __init__(
        self,
        data_dir: str = "data/nport",
        fund_index_path: str = "data/fund_type_index.json",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

        self.filtered_dir = self.data_dir / "filtered"
        self.filtered_dir.mkdir(exist_ok=True)

        self.fund_index_path = Path(fund_index_path)
        self.evergreen_ciks = self._load_evergreen_ciks()

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })

    def _load_evergreen_ciks(self) -> set[str]:
        """Load list of evergreen fund CIKs from the fund type index."""
        if not self.fund_index_path.exists():
            logger.warning(f"Fund index not found at {self.fund_index_path}")
            logger.warning("Run fund_type_indexer first to build the index")
            return set()

        with open(self.fund_index_path) as f:
            index = json.load(f)

        # Combine interval funds and tender offer funds
        interval_ciks = set(index.get("interval_funds", []))
        tender_offer_ciks = set(index.get("tender_offer_funds", []))

        all_ciks = interval_ciks | tender_offer_ciks
        logger.info(f"Loaded {len(all_ciks)} evergreen fund CIKs "
                   f"({len(interval_ciks)} interval, {len(tender_offer_ciks)} tender offer)")

        return all_ciks

    def _get_available_quarters(self) -> list[str]:
        """
        Get list of available N-PORT data quarters.

        Quarters are available from 2019q4 (Oct 2019) onward.
        Format: YYYYqN (e.g., 2025q3)

        Note: SEC bulk data is typically available ~1 quarter behind.
        We include recent quarters and let the download handle 404s.
        """
        quarters = []
        current_year = datetime.now(timezone.utc).year
        current_month = datetime.now(timezone.utc).month
        current_quarter = (current_month - 1) // 3 + 1

        # N-PORT data starts from 2019q4 (October 2019)
        for year in range(2019, current_year + 1):
            start_q = 4 if year == 2019 else 1
            end_q = current_quarter if year == current_year else 4

            for q in range(start_q, end_q + 1):
                quarters.append(f"{year}q{q}")

        # Remove only current quarter (definitely not available yet)
        # Previous quarter may or may not be available depending on timing
        if quarters and quarters[-1] == f"{current_year}q{current_quarter}":
            quarters = quarters[:-1]

        return quarters

    def _get_quarter_url(self, quarter: str) -> str:
        """Get download URL for a specific quarter's data."""
        # URL pattern: https://www.sec.gov/files/dera/data/form-n-port-data-sets/2025q3_nport.zip
        return f"{NPORT_ARCHIVE_BASE}/{quarter}_nport.zip"

    def _download_quarter(self, quarter: str) -> Optional[Path]:
        """Download a quarter's N-PORT data zip file."""
        url = self._get_quarter_url(quarter)
        zip_path = self.raw_dir / f"{quarter}_nport.zip"

        if zip_path.exists():
            logger.info(f"Quarter {quarter} already downloaded: {zip_path}")
            return zip_path

        logger.info(f"Downloading {quarter} from {url}...")

        try:
            time.sleep(REQUEST_DELAY)
            response = self.session.get(url, timeout=300, stream=True)
            response.raise_for_status()

            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) < 8192:  # Log every ~10MB
                            logger.info(f"  Downloaded {downloaded / 1024 / 1024:.1f}MB ({pct:.1f}%)")

            logger.info(f"Downloaded {zip_path.name} ({downloaded / 1024 / 1024:.1f}MB)")
            return zip_path

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Quarter {quarter} not available (404)")
            else:
                logger.error(f"HTTP error downloading {quarter}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {quarter}: {e}")
            return None

    def _extract_table(
        self,
        zip_path: Path,
        table_name: str,
    ) -> Optional[pd.DataFrame]:
        """Extract a specific table from the quarter's zip file."""
        tsv_filename = f"{table_name.lower()}.tsv"

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find the TSV file (may be in a subdirectory)
                matching_files = [
                    name for name in zf.namelist()
                    if name.lower().endswith(tsv_filename)
                ]

                if not matching_files:
                    logger.debug(f"Table {table_name} not found in {zip_path.name}")
                    return None

                tsv_path = matching_files[0]
                logger.debug(f"Reading {tsv_path} from {zip_path.name}")

                with zf.open(tsv_path) as f:
                    # Read TSV with pandas
                    df = pd.read_csv(
                        f,
                        sep='\t',
                        dtype=str,  # Read all as strings initially
                        na_values=['', 'NA', 'N/A'],
                        low_memory=False,
                    )

                logger.debug(f"Loaded {table_name}: {len(df):,} rows")
                return df

        except Exception as e:
            logger.error(f"Error extracting {table_name} from {zip_path.name}: {e}")
            return None

    def _filter_by_cik(
        self,
        df: pd.DataFrame,
        cik_column: str = "CIK",
    ) -> pd.DataFrame:
        """Filter dataframe to only include evergreen fund CIKs."""
        if df is None or df.empty:
            return df

        if cik_column not in df.columns:
            logger.warning(f"CIK column '{cik_column}' not found in dataframe")
            return df

        # Normalize CIKs (remove leading zeros for comparison)
        df_ciks = df[cik_column].astype(str).str.lstrip('0')
        evergreen_ciks_normalized = {cik.lstrip('0') for cik in self.evergreen_ciks}

        mask = df_ciks.isin(evergreen_ciks_normalized)
        filtered_df = df[mask].copy()

        logger.debug(f"Filtered {len(df):,} → {len(filtered_df):,} rows by CIK")
        return filtered_df

    def _filter_holdings_by_accession(
        self,
        holdings_df: pd.DataFrame,
        submission_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter holdings to only include accession numbers from evergreen funds."""
        if holdings_df is None or holdings_df.empty:
            return holdings_df

        if submission_df is None or submission_df.empty:
            return holdings_df

        valid_accessions = set(submission_df["ACCESSION_NUMBER"].unique())
        mask = holdings_df["ACCESSION_NUMBER"].isin(valid_accessions)
        filtered_df = holdings_df[mask].copy()

        logger.debug(f"Filtered holdings {len(holdings_df):,} → {len(filtered_df):,} rows by accession")
        return filtered_df

    def load_quarter(
        self,
        quarter: str,
        tables: Optional[list[str]] = None,
        force_download: bool = False,
    ) -> QuarterData:
        """
        Load a quarter's N-PORT data, filtered to evergreen funds.

        Args:
            quarter: Quarter string (e.g., "2025q3")
            tables: List of tables to load (default: PRIORITY_TABLES)
            force_download: Re-download even if already cached

        Returns:
            QuarterData object with loaded dataframes
        """
        if not self.evergreen_ciks:
            logger.error("No evergreen CIKs loaded. Run fund_type_indexer first.")
            return QuarterData(quarter=quarter)

        tables = tables or PRIORITY_TABLES

        # Check for cached filtered data
        cache_path = self.filtered_dir / quarter
        if cache_path.exists() and not force_download:
            logger.info(f"Loading cached data for {quarter}")
            return self._load_cached_quarter(quarter)

        # Download raw data
        zip_path = self._download_quarter(quarter)
        if zip_path is None:
            return QuarterData(quarter=quarter)

        # Load and filter tables
        data = QuarterData(quarter=quarter)

        # First, load REGISTRANT to get CIK-based filtering
        logger.info(f"Processing {quarter}...")

        # Load submission and registrant first (for filtering)
        submission_df = self._extract_table(zip_path, "SUBMISSION")
        registrant_df = self._extract_table(zip_path, "REGISTRANT")

        if registrant_df is not None:
            # Filter registrant by CIK
            registrant_df = self._filter_by_cik(registrant_df, "CIK")
            data.registrant = registrant_df

            # Get valid accession numbers
            valid_accessions = set(registrant_df["ACCESSION_NUMBER"].unique())

            # Filter submission by valid accessions
            if submission_df is not None:
                submission_df = submission_df[
                    submission_df["ACCESSION_NUMBER"].isin(valid_accessions)
                ].copy()
                data.submission = submission_df

            logger.info(f"Found {len(valid_accessions)} filings from evergreen funds")

            # Load other tables and filter by accession number
            if "FUND_REPORTED_INFO" in tables:
                df = self._extract_table(zip_path, "FUND_REPORTED_INFO")
                if df is not None:
                    df = df[df["ACCESSION_NUMBER"].isin(valid_accessions)].copy()
                    data.fund_reported_info = df
                    logger.info(f"  FUND_REPORTED_INFO: {len(df):,} rows")

            if "MONTHLY_TOTAL_RETURN" in tables:
                df = self._extract_table(zip_path, "MONTHLY_TOTAL_RETURN")
                if df is not None:
                    df = df[df["ACCESSION_NUMBER"].isin(valid_accessions)].copy()
                    data.monthly_total_return = df
                    logger.info(f"  MONTHLY_TOTAL_RETURN: {len(df):,} rows")

            if "FUND_REPORTED_HOLDING" in tables:
                df = self._extract_table(zip_path, "FUND_REPORTED_HOLDING")
                if df is not None:
                    df = df[df["ACCESSION_NUMBER"].isin(valid_accessions)].copy()
                    data.fund_reported_holding = df
                    logger.info(f"  FUND_REPORTED_HOLDING: {len(df):,} rows")

                    # Get valid holding IDs for child tables
                    valid_holding_ids = set(df["HOLDING_ID"].unique())

                    if "IDENTIFIERS" in tables:
                        id_df = self._extract_table(zip_path, "IDENTIFIERS")
                        if id_df is not None:
                            id_df = id_df[id_df["HOLDING_ID"].isin(valid_holding_ids)].copy()
                            data.identifiers = id_df
                            logger.info(f"  IDENTIFIERS: {len(id_df):,} rows")

                    if "DEBT_SECURITY" in tables:
                        debt_df = self._extract_table(zip_path, "DEBT_SECURITY")
                        if debt_df is not None:
                            debt_df = debt_df[debt_df["HOLDING_ID"].isin(valid_holding_ids)].copy()
                            data.debt_security = debt_df
                            logger.info(f"  DEBT_SECURITY: {len(debt_df):,} rows")

        # Cache the filtered data
        self._cache_quarter(data)

        return data

    def _cache_quarter(self, data: QuarterData) -> None:
        """Save filtered quarter data to cache."""
        cache_path = self.filtered_dir / data.quarter
        cache_path.mkdir(exist_ok=True)

        for table_name in ["submission", "registrant", "fund_reported_info",
                          "monthly_total_return", "fund_reported_holding",
                          "identifiers", "debt_security"]:
            df = getattr(data, table_name, None)
            if df is not None and not df.empty:
                parquet_path = cache_path / f"{table_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                logger.debug(f"Cached {table_name} to {parquet_path}")

    def _load_cached_quarter(self, quarter: str) -> QuarterData:
        """Load cached quarter data."""
        cache_path = self.filtered_dir / quarter
        data = QuarterData(quarter=quarter)

        for table_name in ["submission", "registrant", "fund_reported_info",
                          "monthly_total_return", "fund_reported_holding",
                          "identifiers", "debt_security"]:
            parquet_path = cache_path / f"{table_name}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                setattr(data, table_name, df)
                logger.debug(f"Loaded cached {table_name}: {len(df):,} rows")

        return data

    def load_all_quarters(
        self,
        tables: Optional[list[str]] = None,
    ) -> dict[str, QuarterData]:
        """Load all available quarters."""
        quarters = self._get_available_quarters()
        logger.info(f"Loading {len(quarters)} quarters: {quarters[0]} to {quarters[-1]}")

        results = {}
        for quarter in quarters:
            data = self.load_quarter(quarter, tables=tables)
            results[quarter] = data

        return results

    def get_holdings_summary(self, data: QuarterData) -> dict:
        """Generate summary statistics for a quarter's holdings."""
        if data.fund_reported_holding is None:
            return {}

        holdings = data.fund_reported_holding

        summary = {
            "quarter": data.quarter,
            "total_holdings": len(holdings),
            "unique_funds": holdings["ACCESSION_NUMBER"].nunique(),
        }

        # Fair value level distribution
        if "FAIR_VALUE_LEVEL" in holdings.columns:
            fv_dist = holdings["FAIR_VALUE_LEVEL"].value_counts().to_dict()
            summary["fair_value_distribution"] = fv_dist

        # Asset category distribution
        if "ASSET_CAT" in holdings.columns:
            asset_dist = holdings["ASSET_CAT"].value_counts().head(10).to_dict()
            summary["top_asset_categories"] = asset_dist

        # Restricted securities
        if "IS_RESTRICTED_SECURITY" in holdings.columns:
            restricted_count = (holdings["IS_RESTRICTED_SECURITY"] == "Y").sum()
            summary["restricted_securities"] = restricted_count

        return summary


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load SEC N-PORT bulk data for evergreen funds"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Load only the latest available quarter",
    )
    parser.add_argument(
        "--quarter",
        type=str,
        help="Load a specific quarter (e.g., 2025q3)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all available quarters",
    )
    parser.add_argument(
        "--list-quarters",
        action="store_true",
        help="List available quarters",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics",
    )

    args = parser.parse_args()

    loader = NPortLoader()

    if args.list_quarters:
        quarters = loader._get_available_quarters()
        print(f"Available quarters ({len(quarters)}):")
        for q in quarters:
            print(f"  {q}")
        return

    if args.latest:
        quarters = loader._get_available_quarters()
        if quarters:
            quarter = quarters[-1]
            print(f"Loading latest quarter: {quarter}")
            data = loader.load_quarter(quarter)

            if args.summary:
                summary = loader.get_holdings_summary(data)
                print(f"\nSummary for {quarter}:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")

    elif args.quarter:
        print(f"Loading quarter: {args.quarter}")
        data = loader.load_quarter(args.quarter)

        if args.summary:
            summary = loader.get_holdings_summary(data)
            print(f"\nSummary for {args.quarter}:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

    elif args.all:
        print("Loading all available quarters...")
        results = loader.load_all_quarters()
        print(f"\nLoaded {len(results)} quarters")

        if args.summary:
            for quarter, data in results.items():
                summary = loader.get_holdings_summary(data)
                print(f"\n{quarter}: {summary.get('total_holdings', 0):,} holdings")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
