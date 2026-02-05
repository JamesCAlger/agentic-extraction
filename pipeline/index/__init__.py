"""
Fund indexing package.

Provides tools for building and maintaining indexes of SEC-registered funds
by type (interval, tender offer, BDC, etc.).
"""

from .fund_type_indexer import (
    FundTypeIndexer,
    FundIndexEntry,
    FundTypeIndex,
    SECEdgarClient,
)

__all__ = [
    "FundTypeIndexer",
    "FundIndexEntry",
    "FundTypeIndex",
    "SECEdgarClient",
]
