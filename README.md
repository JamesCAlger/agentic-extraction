# Agentic Extraction: LLM-Powered Data Extraction from SEC Filings

A multi-tier agentic extraction pipeline that parses unstructured SEC filings (N-2, N-CSR) into structured, validated data for evergreen private equity funds. Built to solve a real market gap: the $400B+ evergreen fund space lacks specialized data infrastructure despite rich public disclosures.

## The Problem

Evergreen PE funds (interval funds, tender offer funds) file detailed disclosures with the SEC -- fee structures, repurchase terms, incentive fees, leverage limits, allocation targets. This information is public but:

- Buried in 100-200 page HTML/iXBRL documents
- Inconsistently formatted across fund families
- Contains nested tables, cross-references, and legal language
- Requires domain expertise to interpret correctly

Manual extraction doesn't scale. Simple rule-based parsing breaks on format variations. A single LLM call on a 200-page document hallucinates.

## The Approach

A four-tier extraction architecture that progressively escalates from deterministic parsing to agentic LLM search, using each tier only when the previous one fails:

```
Document ──> Tier 0: XBRL Tag Parsing (deterministic, 100% accurate)
         ──> Tier 1: Section-to-Field Mapping + LLM (structured extraction with dynamic few-shot examples)
         ──> Tier 2: Regex Pattern Fallback (finds untagged sections)
         ──> Tier 3: Scoped Agentic Search (keyword scoring -> candidate sections -> focused LLM extraction)
         ──> Grounding Validation (verify every value appears in source text)
         ──> Result Assembly + Evaluation
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| **Tiered fallback** instead of single LLM call | Reduces cost, improves accuracy, maintains determinism where possible |
| **Mandatory source citations** for every extracted value | Enables verification, debugging, and audit trails |
| **Dynamic few-shot examples** based on fund type | Interval funds vs. tender offer funds have different structures |
| **Chain-of-thought reasoning** in extraction schema | Forces the LLM to explain its reasoning before committing to a value |
| **Grounding validation** post-extraction | Catches hallucinations by verifying values appear in source text |
| **Experiment framework** with ground truth evaluation | Systematic A/B testing of every pipeline change |

## Results

Evaluated against manually-verified ground truth across 5 fund families (Blackstone, StepStone, Hamilton Lane, Blue Owl, Carlyle):

| Metric | Baseline (Tier 1 only) | Best Configuration |
|--------|----------------------|-------------------|
| **Accuracy** | 48.4% | **84.7%** |
| **F1 Score** | 0.506 | **0.863** |
| **Precision** | 66.7% | **93.3%** |
| **Recall** | 40.7% | **80.3%** |

Per-fund accuracy (best configuration):

| Fund | Accuracy | Notes |
|------|----------|-------|
| Blackstone | 98.2% | Tender offer fund, large document |
| Hamilton Lane | 84.0% | Tender offer fund, fund-level incentive fee |
| StepStone | 82.0% | Interval fund, 2800+ chunks, fund-of-funds |
| Blue Owl | 80.0% | Direct credit fund, different terminology |
| Carlyle AlpInvest | 78.0% | Secondaries fund |

**Best configuration**: Tier 3 keyword search (top_k=10) + Cohere Reranker (rerank-v3.5) + Tier 4 escalation (Claude Haiku) + post-extraction validation rules.

## Architecture

### Extraction Pipeline

```
SEC EDGAR ──> Fetch & Archive ──> Document Segmentation ──> iXBRL Parsing
                                                        ──> Section-to-Field Mapping
                                                        ──> LLM Extraction (instructor + Pydantic)
                                                        ──> Tiered Fallback Search
                                                        ──> Grounding Validation
                                                        ──> Evaluation vs. Ground Truth
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Extraction Pipeline | Python 3.11+, instructor, Pydantic |
| LLM Providers | OpenAI GPT-4o-mini, Claude Haiku, Gemini Flash |
| Retrieval | BM25 keyword search, Cohere Reranker, sentence-transformers |
| Experiment Framework | Custom YAML-based config, automated evaluation |
| Data Store | PostgreSQL (Supabase) + pgvector |
| Orchestration | Dagster |

### Data Points Extracted (per fund)

50+ fields across these categories:
- **Fund classification** -- interval fund vs. tender offer, strategy type
- **Share classes** -- class name, minimum investment, sales load, distribution fees
- **Incentive fees** -- fee percentage, hurdle rate, high water mark, catch-up provisions
- **Repurchase terms** -- frequency, percentage range, lock-up period, early repurchase fees
- **Leverage limits** -- maximum leverage, basis (total assets vs. net assets)
- **Distribution terms** -- frequency, default reinvestment policy
- **Allocation targets** -- secondary funds, direct investments, co-investments
- **Concentration limits** -- max single asset, fund, sector exposure

## Repository Structure

```
agentic-extraction/
├── pipeline/
│   ├── extract/          # Core extraction logic
│   │   ├── extractor.py          # Main extraction orchestrator
│   │   ├── scoped_agentic.py     # Tier 3: keyword-guided agentic search
│   │   ├── tier4_agentic.py      # Tier 4: full agentic escalation
│   │   ├── ensemble_t4_extractor.py  # Multi-model ensemble extraction
│   │   ├── grounding.py          # Source text validation
│   │   ├── schemas.py            # Pydantic schemas with chain-of-thought
│   │   ├── examples.py           # Dynamic few-shot example selection
│   │   ├── prompts.py            # Section-to-field mapping definitions
│   │   ├── validation_rules.py   # Post-extraction disambiguation
│   │   └── ...                   # Additional extraction strategies
│   ├── parse/            # Document preprocessing
│   │   ├── ixbrl_parser.py       # Deterministic XBRL tag extraction
│   │   ├── document_segmenter.py # Section identification
│   │   ├── chunker.py            # Token-aware document chunking
│   │   └── ...
│   ├── experiment/       # Experiment framework
│   │   ├── runner.py             # Run extraction experiments
│   │   ├── evaluator.py          # Precision/recall/F1 evaluation
│   │   └── config.py             # YAML-based experiment configuration
│   ├── validate/         # Validation harness
│   ├── review/           # Human review tools (Streamlit)
│   ├── nport/            # N-PORT XML parser
│   └── index/            # Fund type classification
├── configs/
│   ├── base.yaml                 # Default pipeline configuration
│   ├── experiments/              # Experiment configuration variants
│   └── ground_truth/             # Manually verified extraction targets
├── data/
│   ├── examples/                 # Few-shot extraction examples
│   └── experiments/              # Baseline + best experiment results
├── scripts/              # Pipeline execution scripts
├── tests/                # Unit and integration tests
└── docs/                 # Technical documentation
```

## Experimentation Methodology

Every pipeline change is validated through systematic A/B testing:

1. **Ground truth creation** -- Manually verify 50 fields per fund across 5 fund families
2. **Baseline measurement** -- Run extraction with current best configuration
3. **Variant testing** -- Change one variable (chunking strategy, LLM model, retrieval method, etc.)
4. **Automated evaluation** -- Compare precision, recall, F1, accuracy per fund and per field
5. **Error analysis** -- Categorize failures as wrong value, missed field, or hallucination

The `data/experiments/` directory contains the baseline (48.4% accuracy) and best result (84.7% accuracy) with full evaluation breakdowns.

## Status

This is a POC / proof of concept. The extraction pipeline is functional and evaluated. Remaining work:

- **Scale validation** to additional fund families beyond the initial 5
- **API and UI** for querying extracted data (Phase 2, blocked on 95%+ accuracy target)
- **N-CSR extraction** for financial data (annual reports vs. registration statements)
- **Entity resolution** for cross-fund holdings comparison
