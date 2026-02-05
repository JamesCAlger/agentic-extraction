"""
N-PORT Database Loader

Loads N-PORT data into DuckDB for analysis.
Schema designed for easy migration to PostgreSQL.

Usage:
    python -m pipeline.nport.database --load-all
    python -m pipeline.nport.database --query "SELECT * FROM funds LIMIT 10"
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .loader import NPortLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Schema designed for PostgreSQL compatibility
SCHEMA_SQL = """
-- Fund registry (from our fund type index)
CREATE TABLE IF NOT EXISTS funds (
    cik VARCHAR PRIMARY KEY,
    name VARCHAR,
    fund_type VARCHAR,  -- interval_fund, tender_offer_fund, bdc
    lei VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quarterly filing submissions
CREATE TABLE IF NOT EXISTS submissions (
    accession_number VARCHAR PRIMARY KEY,
    cik VARCHAR,
    filing_date DATE,
    report_date DATE,
    quarter VARCHAR,  -- e.g., "2025q3"
    FOREIGN KEY (cik) REFERENCES funds(cik)
);

-- Fund-level data per filing
CREATE TABLE IF NOT EXISTS fund_periods (
    accession_number VARCHAR PRIMARY KEY,
    cik VARCHAR,
    series_name VARCHAR,
    series_id VARCHAR,
    series_lei VARCHAR,
    quarter VARCHAR,
    report_date DATE,

    -- Assets & Liabilities
    total_assets DECIMAL(18,2),
    total_liabilities DECIMAL(18,2),
    net_assets DECIMAL(18,2),

    -- Flows (quarterly totals)
    sales_flow_mon1 DECIMAL(18,2),
    sales_flow_mon2 DECIMAL(18,2),
    sales_flow_mon3 DECIMAL(18,2),
    redemption_flow_mon1 DECIMAL(18,2),
    redemption_flow_mon2 DECIMAL(18,2),
    redemption_flow_mon3 DECIMAL(18,2),
    reinvestment_flow_mon1 DECIMAL(18,2),
    reinvestment_flow_mon2 DECIMAL(18,2),
    reinvestment_flow_mon3 DECIMAL(18,2),

    -- Computed flows
    gross_sales DECIMAL(18,2),
    gross_redemptions DECIMAL(18,2),
    net_flows DECIMAL(18,2),

    -- Credit spread risk
    credit_spread_3mon_invest DECIMAL(18,6),
    credit_spread_1yr_invest DECIMAL(18,6),
    credit_spread_5yr_invest DECIMAL(18,6),
    credit_spread_10yr_invest DECIMAL(18,6),

    -- Realized/Unrealized gains
    net_realize_gain_nonderiv_mon1 DECIMAL(18,2),
    net_realize_gain_nonderiv_mon2 DECIMAL(18,2),
    net_realize_gain_nonderiv_mon3 DECIMAL(18,2),
    net_unrealize_ap_nonderiv_mon1 DECIMAL(18,2),
    net_unrealize_ap_nonderiv_mon2 DECIMAL(18,2),
    net_unrealize_ap_nonderiv_mon3 DECIMAL(18,2),

    FOREIGN KEY (cik) REFERENCES funds(cik)
);

-- Monthly returns by share class
CREATE TABLE IF NOT EXISTS monthly_returns (
    id INTEGER,
    accession_number VARCHAR,
    class_id VARCHAR,
    monthly_total_return1 DECIMAL(10,6),
    monthly_total_return2 DECIMAL(10,6),
    monthly_total_return3 DECIMAL(10,6),
    quarter VARCHAR,
    PRIMARY KEY (accession_number, id)
);

-- Individual holdings
CREATE TABLE IF NOT EXISTS holdings (
    holding_id BIGINT,
    accession_number VARCHAR,
    quarter VARCHAR,

    -- Identification
    issuer_name VARCHAR,
    issuer_lei VARCHAR,
    issuer_title VARCHAR,
    issuer_cusip VARCHAR(9),

    -- Valuation
    balance DECIMAL(18,4),
    unit VARCHAR,
    currency_code VARCHAR(3),
    currency_value DECIMAL(18,2),
    exchange_rate DECIMAL(12,6),
    percentage DECIMAL(10,6),

    -- Classification
    payoff_profile VARCHAR,
    asset_cat VARCHAR,
    issuer_type VARCHAR,
    investment_country VARCHAR(10),
    is_restricted_security CHAR(1),
    fair_value_level VARCHAR(2),
    derivative_cat VARCHAR,

    PRIMARY KEY (accession_number, holding_id)
);

-- Additional identifiers for holdings
CREATE TABLE IF NOT EXISTS holding_identifiers (
    holding_id BIGINT,
    identifiers_id BIGINT,
    identifier_isin VARCHAR(12),
    identifier_ticker VARCHAR(50),
    other_identifier VARCHAR,
    other_identifier_desc VARCHAR,
    PRIMARY KEY (holding_id, identifiers_id)
);

-- Debt security details
CREATE TABLE IF NOT EXISTS debt_securities (
    holding_id BIGINT PRIMARY KEY,
    maturity_date DATE,
    coupon_type VARCHAR,
    annualized_rate DECIMAL(10,6),
    is_default CHAR(1),
    are_any_interest_payment CHAR(1),
    is_any_portion_interest_paid CHAR(1),
    is_convtible_mandatory CHAR(1),
    is_convtible_contingent CHAR(1)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_holdings_accession ON holdings(accession_number);
CREATE INDEX IF NOT EXISTS idx_holdings_fair_value ON holdings(fair_value_level);
CREATE INDEX IF NOT EXISTS idx_holdings_asset_cat ON holdings(asset_cat);
CREATE INDEX IF NOT EXISTS idx_fund_periods_quarter ON fund_periods(quarter);
CREATE INDEX IF NOT EXISTS idx_fund_periods_cik ON fund_periods(cik);
CREATE INDEX IF NOT EXISTS idx_submissions_quarter ON submissions(quarter);

-- Materialized view equivalent: Fund summary per quarter
CREATE OR REPLACE VIEW fund_quarter_summary AS
SELECT
    fp.cik,
    f.name as fund_name,
    f.fund_type,
    fp.quarter,
    fp.report_date,
    fp.net_assets,
    fp.gross_sales,
    fp.gross_redemptions,
    fp.net_flows,
    COUNT(DISTINCT h.holding_id) as holdings_count,
    SUM(CASE WHEN h.fair_value_level = '1' THEN h.currency_value ELSE 0 END) as level1_value,
    SUM(CASE WHEN h.fair_value_level = '2' THEN h.currency_value ELSE 0 END) as level2_value,
    SUM(CASE WHEN h.fair_value_level = '3' THEN h.currency_value ELSE 0 END) as level3_value,
    SUM(CASE WHEN h.is_restricted_security = 'Y' THEN h.currency_value ELSE 0 END) as restricted_value
FROM fund_periods fp
JOIN funds f ON fp.cik = f.cik
LEFT JOIN holdings h ON fp.accession_number = h.accession_number
GROUP BY fp.cik, f.name, f.fund_type, fp.quarter, fp.report_date,
         fp.net_assets, fp.gross_sales, fp.gross_redemptions, fp.net_flows;
"""


class NPortDatabase:
    """DuckDB database for N-PORT data analysis."""

    def __init__(
        self,
        db_path: str = "data/nport/nport.duckdb",
        fund_index_path: str = "data/fund_type_index.json",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fund_index_path = Path(fund_index_path)

        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        # DuckDB doesn't have executescript, so execute statements individually
        statements = SCHEMA_SQL.split(';')
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and not stmt.startswith('--'):
                try:
                    self.conn.execute(stmt)
                except Exception as e:
                    # Ignore errors for CREATE INDEX IF NOT EXISTS on existing indexes
                    if 'already exists' not in str(e).lower():
                        logger.debug(f"Schema statement warning: {e}")
        logger.info(f"Database initialized at {self.db_path}")

    def _load_fund_index(self):
        """Load fund type index into funds table."""
        if not self.fund_index_path.exists():
            logger.warning("Fund index not found")
            return

        with open(self.fund_index_path) as f:
            index = json.load(f)

        funds_data = []
        for cik, fund in index.get("funds", {}).items():
            funds_data.append({
                "cik": cik,
                "name": fund.get("name"),
                "fund_type": fund.get("fund_type"),
                "lei": fund.get("lei"),
            })

        if funds_data:
            df = pd.DataFrame(funds_data)
            # Upsert logic - delete existing and insert
            self.conn.execute("DELETE FROM funds WHERE cik IN (SELECT cik FROM funds)")
            self.conn.execute("INSERT INTO funds SELECT * FROM df")
            logger.info(f"Loaded {len(funds_data)} funds into database")

    def load_quarter(self, quarter: str, data_dir: str = "data/nport/filtered"):
        """Load a quarter's data from cached parquet files."""
        cache_path = Path(data_dir) / quarter

        if not cache_path.exists():
            logger.warning(f"No cached data for {quarter} at {cache_path}")
            return False

        logger.info(f"Loading {quarter} into database...")

        # Load registrant data
        registrant_path = cache_path / "registrant.parquet"
        if registrant_path.exists():
            df = pd.read_parquet(registrant_path)
            df['quarter'] = quarter

            # Update funds table with any new funds
            for _, row in df.iterrows():
                self.conn.execute("""
                    INSERT INTO funds (cik, name, lei)
                    VALUES (?, ?, ?)
                    ON CONFLICT (cik) DO UPDATE SET
                        name = EXCLUDED.name,
                        lei = COALESCE(EXCLUDED.lei, funds.lei)
                """, [row['CIK'], row['REGISTRANT_NAME'], row.get('LEI')])

        # Load submission data
        submission_path = cache_path / "submission.parquet"
        if submission_path.exists():
            df = pd.read_parquet(submission_path)
            df['quarter'] = quarter

            # Delete existing submissions for this quarter to avoid duplicates
            self.conn.execute("DELETE FROM submissions WHERE quarter = ?", [quarter])

            insert_df = df[['ACCESSION_NUMBER', 'FILING_DATE', 'REPORT_DATE']].copy()
            insert_df.columns = ['accession_number', 'filing_date', 'report_date']
            insert_df['quarter'] = quarter

            # Get CIK from registrant
            reg_df = pd.read_parquet(registrant_path) if registrant_path.exists() else None
            if reg_df is not None:
                cik_map = dict(zip(reg_df['ACCESSION_NUMBER'], reg_df['CIK']))
                insert_df['cik'] = insert_df['accession_number'].map(cik_map)

            self.conn.execute("INSERT INTO submissions SELECT * FROM insert_df")
            logger.info(f"  Loaded {len(insert_df)} submissions")

        # Load fund_reported_info
        fund_info_path = cache_path / "fund_reported_info.parquet"
        if fund_info_path.exists():
            df = pd.read_parquet(fund_info_path)
            df['quarter'] = quarter

            # Get CIK mapping
            if registrant_path.exists():
                reg_df = pd.read_parquet(registrant_path)
                cik_map = dict(zip(reg_df['ACCESSION_NUMBER'], reg_df['CIK']))
                df['CIK'] = df['ACCESSION_NUMBER'].map(cik_map)

            # Calculate aggregate flows
            for col in ['SALES_FLOW_MON1', 'SALES_FLOW_MON2', 'SALES_FLOW_MON3',
                       'REDEMPTION_FLOW_MON1', 'REDEMPTION_FLOW_MON2', 'REDEMPTION_FLOW_MON3']:
                df[col] = pd.to_numeric(df.get(col), errors='coerce').fillna(0)

            df['gross_sales'] = df['SALES_FLOW_MON1'] + df['SALES_FLOW_MON2'] + df['SALES_FLOW_MON3']
            df['gross_redemptions'] = df['REDEMPTION_FLOW_MON1'] + df['REDEMPTION_FLOW_MON2'] + df['REDEMPTION_FLOW_MON3']
            df['net_flows'] = df['gross_sales'] - df['gross_redemptions']

            # Delete existing and insert
            self.conn.execute("DELETE FROM fund_periods WHERE quarter = ?", [quarter])

            # Map columns
            col_mapping = {
                'ACCESSION_NUMBER': 'accession_number',
                'CIK': 'cik',
                'SERIES_NAME': 'series_name',
                'SERIES_ID': 'series_id',
                'SERIES_LEI': 'series_lei',
                'TOTAL_ASSETS': 'total_assets',
                'TOTAL_LIABILITIES': 'total_liabilities',
                'NET_ASSETS': 'net_assets',
                'SALES_FLOW_MON1': 'sales_flow_mon1',
                'SALES_FLOW_MON2': 'sales_flow_mon2',
                'SALES_FLOW_MON3': 'sales_flow_mon3',
                'REDEMPTION_FLOW_MON1': 'redemption_flow_mon1',
                'REDEMPTION_FLOW_MON2': 'redemption_flow_mon2',
                'REDEMPTION_FLOW_MON3': 'redemption_flow_mon3',
            }

            insert_df = df.rename(columns=col_mapping)
            insert_df['quarter'] = quarter

            # Select only columns that exist
            valid_cols = [c for c in ['accession_number', 'cik', 'series_name', 'series_id',
                                      'series_lei', 'quarter', 'total_assets', 'total_liabilities',
                                      'net_assets', 'sales_flow_mon1', 'sales_flow_mon2', 'sales_flow_mon3',
                                      'redemption_flow_mon1', 'redemption_flow_mon2', 'redemption_flow_mon3',
                                      'gross_sales', 'gross_redemptions', 'net_flows']
                         if c in insert_df.columns]
            insert_df = insert_df[valid_cols]

            self.conn.execute(f"INSERT INTO fund_periods ({','.join(valid_cols)}) SELECT {','.join(valid_cols)} FROM insert_df")
            logger.info(f"  Loaded {len(insert_df)} fund periods")

        # Load monthly returns
        returns_path = cache_path / "monthly_total_return.parquet"
        if returns_path.exists():
            df = pd.read_parquet(returns_path)
            df['quarter'] = quarter

            self.conn.execute("DELETE FROM monthly_returns WHERE quarter = ?", [quarter])

            insert_df = df.rename(columns={
                'ACCESSION_NUMBER': 'accession_number',
                'MONTHLY_TOTAL_RETURN_ID': 'id',
                'CLASS_ID': 'class_id',
                'MONTHLY_TOTAL_RETURN1': 'monthly_total_return1',
                'MONTHLY_TOTAL_RETURN2': 'monthly_total_return2',
                'MONTHLY_TOTAL_RETURN3': 'monthly_total_return3',
            })
            insert_df['quarter'] = quarter

            valid_cols = ['id', 'accession_number', 'class_id', 'monthly_total_return1',
                         'monthly_total_return2', 'monthly_total_return3', 'quarter']
            valid_cols = [c for c in valid_cols if c in insert_df.columns]
            insert_df = insert_df[valid_cols]

            self.conn.execute(f"INSERT INTO monthly_returns ({','.join(valid_cols)}) SELECT {','.join(valid_cols)} FROM insert_df")
            logger.info(f"  Loaded {len(insert_df)} return records")

        # Load holdings (largest table)
        holdings_path = cache_path / "fund_reported_holding.parquet"
        if holdings_path.exists():
            df = pd.read_parquet(holdings_path)
            df['quarter'] = quarter

            self.conn.execute("DELETE FROM holdings WHERE quarter = ?", [quarter])

            col_mapping = {
                'HOLDING_ID': 'holding_id',
                'ACCESSION_NUMBER': 'accession_number',
                'ISSUER_NAME': 'issuer_name',
                'ISSUER_LEI': 'issuer_lei',
                'ISSUER_TITLE': 'issuer_title',
                'ISSUER_CUSIP': 'issuer_cusip',
                'BALANCE': 'balance',
                'UNIT': 'unit',
                'CURRENCY_CODE': 'currency_code',
                'CURRENCY_VALUE': 'currency_value',
                'EXCHANGE_RATE': 'exchange_rate',
                'PERCENTAGE': 'percentage',
                'PAYOFF_PROFILE': 'payoff_profile',
                'ASSET_CAT': 'asset_cat',
                'ISSUER_TYPE': 'issuer_type',
                'INVESTMENT_COUNTRY': 'investment_country',
                'IS_RESTRICTED_SECURITY': 'is_restricted_security',
                'FAIR_VALUE_LEVEL': 'fair_value_level',
                'DERIVATIVE_CAT': 'derivative_cat',
            }

            insert_df = df.rename(columns=col_mapping)
            insert_df['quarter'] = quarter

            valid_cols = [c for c in col_mapping.values() if c in insert_df.columns] + ['quarter']
            insert_df = insert_df[valid_cols]

            # Convert numeric columns
            for col in ['balance', 'currency_value', 'exchange_rate', 'percentage']:
                if col in insert_df.columns:
                    insert_df[col] = pd.to_numeric(insert_df[col], errors='coerce')

            self.conn.execute(f"INSERT INTO holdings ({','.join(valid_cols)}) SELECT {','.join(valid_cols)} FROM insert_df")
            logger.info(f"  Loaded {len(insert_df):,} holdings")

        # Load identifiers
        identifiers_path = cache_path / "identifiers.parquet"
        if identifiers_path.exists():
            df = pd.read_parquet(identifiers_path)

            # Get holding IDs that exist in this quarter
            existing_holdings = self.conn.execute(
                "SELECT DISTINCT holding_id FROM holdings WHERE quarter = ?", [quarter]
            ).fetchdf()

            if len(existing_holdings) > 0:
                df = df[df['HOLDING_ID'].isin(existing_holdings['holding_id'])]

                self.conn.execute("""
                    DELETE FROM holding_identifiers
                    WHERE holding_id IN (SELECT holding_id FROM holdings WHERE quarter = ?)
                """, [quarter])

                insert_df = df.rename(columns={
                    'HOLDING_ID': 'holding_id',
                    'IDENTIFIERS_ID': 'identifiers_id',
                    'IDENTIFIER_ISIN': 'identifier_isin',
                    'IDENTIFIER_TICKER': 'identifier_ticker',
                    'OTHER_IDENTIFIER': 'other_identifier',
                    'OTHER_IDENTIFIER_DESC': 'other_identifier_desc',
                })

                valid_cols = ['holding_id', 'identifiers_id', 'identifier_isin',
                             'identifier_ticker', 'other_identifier', 'other_identifier_desc']
                valid_cols = [c for c in valid_cols if c in insert_df.columns]
                insert_df = insert_df[valid_cols]

                self.conn.execute(f"INSERT INTO holding_identifiers ({','.join(valid_cols)}) SELECT {','.join(valid_cols)} FROM insert_df")
                logger.info(f"  Loaded {len(insert_df):,} identifiers")

        # Load debt securities
        debt_path = cache_path / "debt_security.parquet"
        if debt_path.exists():
            df = pd.read_parquet(debt_path)

            # Get holding IDs that exist
            existing_holdings = self.conn.execute(
                "SELECT DISTINCT holding_id FROM holdings WHERE quarter = ?", [quarter]
            ).fetchdf()

            if len(existing_holdings) > 0:
                df = df[df['HOLDING_ID'].isin(existing_holdings['holding_id'])]

                self.conn.execute("""
                    DELETE FROM debt_securities
                    WHERE holding_id IN (SELECT holding_id FROM holdings WHERE quarter = ?)
                """, [quarter])

                insert_df = df.rename(columns={
                    'HOLDING_ID': 'holding_id',
                    'MATURITY_DATE': 'maturity_date',
                    'COUPON_TYPE': 'coupon_type',
                    'ANNUALIZED_RATE': 'annualized_rate',
                    'IS_DEFAULT': 'is_default',
                    'ARE_ANY_INTEREST_PAYMENT': 'are_any_interest_payment',
                    'IS_ANY_PORTION_INTEREST_PAID': 'is_any_portion_interest_paid',
                    'IS_CONVTIBLE_MANDATORY': 'is_convtible_mandatory',
                    'IS_CONVTIBLE_CONTINGENT': 'is_convtible_contingent',
                })

                valid_cols = ['holding_id', 'maturity_date', 'coupon_type', 'annualized_rate',
                             'is_default', 'are_any_interest_payment']
                valid_cols = [c for c in valid_cols if c in insert_df.columns]
                insert_df = insert_df[valid_cols]

                # Convert rate to numeric
                if 'annualized_rate' in insert_df.columns:
                    insert_df['annualized_rate'] = pd.to_numeric(insert_df['annualized_rate'], errors='coerce')

                self.conn.execute(f"INSERT INTO debt_securities ({','.join(valid_cols)}) SELECT {','.join(valid_cols)} FROM insert_df")
                logger.info(f"  Loaded {len(insert_df):,} debt securities")

        return True

    def load_all_quarters(self, data_dir: str = "data/nport/filtered"):
        """Load all available quarters from cache."""
        cache_dir = Path(data_dir)

        if not cache_dir.exists():
            logger.error(f"Cache directory not found: {cache_dir}")
            return

        # Find all quarter directories
        quarters = sorted([d.name for d in cache_dir.iterdir() if d.is_dir() and 'q' in d.name])

        if not quarters:
            logger.warning("No cached quarters found. Run NPortLoader first.")
            return

        logger.info(f"Found {len(quarters)} quarters to load: {quarters[0]} to {quarters[-1]}")

        # Load fund index first
        self._load_fund_index()

        # Load each quarter
        for quarter in quarters:
            self.load_quarter(quarter, data_dir)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print database summary."""
        print("\n" + "=" * 60)
        print("DATABASE SUMMARY")
        print("=" * 60)

        tables = ['funds', 'submissions', 'fund_periods', 'monthly_returns', 'holdings',
                  'holding_identifiers', 'debt_securities']

        for table in tables:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {count:,} rows")
            except Exception as e:
                print(f"  {table}: Error - {e}")

        # Quarters loaded
        quarters = self.conn.execute(
            "SELECT DISTINCT quarter FROM fund_periods ORDER BY quarter"
        ).fetchdf()
        print(f"\nQuarters loaded: {len(quarters)}")
        if len(quarters) > 0:
            print(f"  Range: {quarters.iloc[0]['quarter']} to {quarters.iloc[-1]['quarter']}")

        # Holdings by fair value level
        fv_dist = self.conn.execute("""
            SELECT fair_value_level, COUNT(*) as count,
                   SUM(currency_value) as total_value
            FROM holdings
            GROUP BY fair_value_level
            ORDER BY fair_value_level
        """).fetchdf()
        print("\nHoldings by Fair Value Level:")
        for _, row in fv_dist.iterrows():
            level = row['fair_value_level'] or 'NULL'
            count = row['count']
            value = row['total_value'] or 0
            print(f"  Level {level}: {count:,} holdings (${value/1e9:.1f}B)")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="N-PORT Database Operations")
    parser.add_argument("--load-all", action="store_true", help="Load all cached quarters")
    parser.add_argument("--load-quarter", type=str, help="Load specific quarter")
    parser.add_argument("--query", type=str, help="Execute SQL query")
    parser.add_argument("--summary", action="store_true", help="Print database summary")

    args = parser.parse_args()

    db = NPortDatabase()

    try:
        if args.load_all:
            # First download all quarters if not cached
            loader = NPortLoader()
            quarters = loader._get_available_quarters()

            print(f"Downloading/caching {len(quarters)} quarters...")
            for quarter in quarters:
                cache_path = Path(f"data/nport/filtered/{quarter}")
                if not cache_path.exists():
                    print(f"  Downloading {quarter}...")
                    loader.load_quarter(quarter)
                else:
                    print(f"  {quarter} already cached")

            # Then load into database
            db.load_all_quarters()

        elif args.load_quarter:
            db.load_quarter(args.load_quarter)

        elif args.query:
            result = db.query(args.query)
            print(result.to_string())

        elif args.summary:
            db._print_summary()

        else:
            parser.print_help()

    finally:
        db.close()


if __name__ == "__main__":
    main()
