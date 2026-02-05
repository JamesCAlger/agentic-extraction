# LLM Extraction Architecture

This document describes the architecture for extracting narrative fields from SEC N-2 and N-CSR filings using LLM-based structured extraction.

---

## Overview

The extraction system uses a hybrid approach:

1. **Deterministic XBRL Extraction** - Numeric values (fees, expense ratios) are extracted directly from iXBRL tags without LLM involvement
2. **LLM Narrative Extraction** - Complex fields embedded in prose (repurchase terms, allocation targets) are extracted using GPT-4o-mini with structured output schemas
3. **Fallback Pattern Search** - When tagged sections don't contain target fields, regex-based section finding locates content in untagged HTML

This approach minimizes LLM costs while ensuring accuracy for structured fields.

---

## Current Implementation Status

| Component | Status | Accuracy |
|-----------|--------|----------|
| XBRL Tag Parsing | Implemented | 100% (deterministic) |
| Section-to-Field Mapping | Implemented | ~80% coverage |
| Fallback Pattern Search | Implemented | Catches ~15% more fields |
| LLM Structured Extraction | Implemented | ~71% field extraction |
| Agentic Search Fallback | **Not Yet Implemented** | Next priority |

### Extraction Results (Validated on StepStone & Blackstone N-2s)

| Field | StepStone | Blackstone |
|-------|-----------|------------|
| Incentive Fee | EXTRACTED | N/A (no fund-level fee) |
| Expense Cap | N/A (none exists) | EXTRACTED |
| Repurchase Terms | EXTRACTED | EXTRACTED |
| Allocation Targets | EXTRACTED | EXTRACTED |
| Concentration Limits | NOT FOUND | EXTRACTED |
| Share Classes | EXTRACTED | NOT FOUND |

---

## Directory Structure

| Path | Description |
|------|-------------|
| `pipeline/extract/schemas.py` | Pydantic models defining extraction output structure |
| `pipeline/extract/prompts.py` | LLM prompts and section-to-field mapping |
| `pipeline/extract/extractor.py` | Main extraction orchestration logic |
| `pipeline/extract/section_finder.py` | Fallback regex-based section search |
| `pipeline/parse/` | Document parsing and chunking (upstream dependency) |
| `data/extracted/` | Output location for extraction results |

---

## Section-to-Field Mapping

Based on analysis of StepStone (CIK 0001789470) and Blackstone (CIK 0002032432) N-2 filings:

### XBRL-Tagged Sections (Deterministic Extraction)

| Field | XBRL Tag | Notes |
|-------|----------|-------|
| Management Fee | `cef:ManagementFeesPercent` | By share class via contextRef |
| Sales Load | `cef:SalesLoadPercent` | By share class |
| Distribution Fee | `cef:DistributionServicingFeesPercent` | By share class |
| AFFE | `cef:AcquiredFundFeesAndExpensesPercent` | By share class |
| Total Expense Ratio | `cef:TotalAnnualExpensesPercent` | By share class |
| Interest Expense | `cef:InterestExpensesOnBorrowingsPercent` | By share class |
| Expense Example | `cef:ExpenseExampleYear01`, etc. | 1, 3, 5, 10 year projections |

### Narrative Sections (LLM Extraction Required)

| Field | Primary Source Section(s) | Fallback Patterns |
|-------|---------------------------|-------------------|
| Incentive Fee | "Acquired Fund Fees And Expenses Note" | N/A |
| Expense Cap | "Other Expenses Note", fee waiver disclosures | N/A |
| Repurchase Terms | (Untagged) | `REPURCHASES?\s+OF\s+SHARES`, `TENDER\s+OFFERS?` |
| Allocation Targets | "Investment Objectives And Practices" | N/A |
| Concentration Limits | "Investment Objectives And Practices" | `INVESTMENT\s+RESTRICTIONS`, `FUNDAMENTAL\s+POLICIES` |
| Share Class Details | (Untagged) | `DESCRIPTION\s+OF\s+SHARES`, `PLAN\s+OF\s+DISTRIBUTION` |

---

## Extraction Pipeline Flow

### Current Implementation (Tier 1 + Fallback)

```
Document → XBRL Parser → Section Segmenter → Chunk Builder
                                    ↓
                         Section-to-Field Mapping
                                    ↓
                    LLM Extraction (per field, per chunk)
                                    ↓
                         Missing Fields Check
                                    ↓
                    Fallback Pattern Search (section_finder.py)
                                    ↓
                    LLM Extraction on Fallback Content
                                    ↓
                         Result Assembly + Validation
```

### Stage 1: XBRL + Section Mapping

1. Parse iXBRL tags for numeric fees (deterministic)
2. Segment document into sections by XBRL text blocks and HTML headings
3. Map sections to target extraction fields
4. Chunk sections for LLM token limits

### Stage 2: LLM Extraction

For each field with relevant chunks:
1. Combine chunk content (up to token limit)
2. Apply field-specific prompt from `prompts.py`
3. Call LLM with structured output schema via instructor library
4. Validate response against Pydantic schema

### Stage 3: Fallback Search

For fields that returned null or NOT_FOUND:
1. Search full document text using regex patterns (section_finder.py)
2. Extract matching section content up to max_chars limit
3. Run LLM extraction on fallback content
4. Merge with primary results

### Stage 4: Result Assembly

Combine:
- XBRL numeric values (deterministic, by share class)
- LLM narrative extractions (with citations)
- Fallback extractions (with "Fallback: {field}" citation)
- Processing metadata (chunks processed, errors)

---

## Extraction Schemas

Defined in `pipeline/extract/schemas.py`:

| Schema | Purpose | Key Fields |
|--------|---------|------------|
| `IncentiveFeeExtraction` | Performance-based fees | has_incentive_fee, rate, hurdle_rate, high_water_mark |
| `ExpenseCapExtraction` | Fee waivers | has_expense_cap, cap_rate, expiration, recoupment |
| `RepurchaseTermsExtraction` | Liquidity terms | frequency, percentage range, notice period, early fees |
| `AllocationTargetsExtraction` | Asset allocation | list of targets with percentages/ranges |
| `ConcentrationLimitsExtraction` | Investment limits | list of limits by type (issuer, sector, etc.) |
| `ShareClassesExtraction` | Class details | list of classes with minimums and eligibility |

All schemas include:
- `confidence` field (explicit/inferred/not_found)
- `citation` field with evidence quote from source text

### Schema Design Notes

- Use `Optional[str]` for percentage fields to handle LLM returning "not_found" or descriptive values like "at least 80%"
- Decimal types cause validation errors when LLM returns non-numeric strings

---

## Prompt Design

Prompts are defined in `pipeline/extract/prompts.py`.

### System Prompt

Establishes the LLM as a financial document analyst with instructions on:
- Accuracy requirements
- Citation obligations
- Confidence level definitions
- Fund-of-funds fee distinction (fund-level vs underlying)

### Field-Specific Prompts

Each target field has a dedicated prompt that includes:
- What to look for
- Common patterns in SEC filings
- Examples of relevant language
- Guidance on edge cases
- **Explicit exclusions** (e.g., "Interest on borrowings is NOT an incentive fee")

---

## Confidence Levels

| Level | Definition | When to Use |
|-------|------------|-------------|
| `explicit` | Value directly stated in text | "quarterly repurchase offer for 5%" |
| `inferred` | Value derived from calculation or implication | Calculated from multiple stated values |
| `not_found` | Value not present in provided text | Section doesn't contain target information |

---

## Error Handling

The extraction system handles:
- Missing sections (returns null with not_found confidence)
- LLM parsing failures (logged, included in extraction_errors list)
- Token limit exceeded (chunks are combined up to limit, excess dropped)
- Schema validation failures (instructor retries with correction prompt)
- Fallback content too short (skipped if < 500 chars)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `instructor` | Structured LLM output with Pydantic validation |
| `openai` | LLM API client |
| `pydantic` | Schema definitions and validation |
| `tiktoken` | Token counting for chunk management |
| `beautifulsoup4` | HTML parsing for fallback section finder |

---

## Next Steps: Agentic Search (Tier 2)

The next enhancement is an **agentic search fallback** that uses chain-of-thought reasoning and few-shot examples to improve extraction accuracy when Tier 1 fails.

### Planned Architecture

```
Tier 1 Result: NOT_FOUND or Low Confidence
                    ↓
            Agentic Search Agent
                    ↓
    ┌───────────────────────────────────┐
    │  1. Chain-of-Thought Reasoning    │
    │     "I need to find X. Likely     │
    │      locations are A, B, C..."    │
    │                                   │
    │  2. Section Search Tool           │
    │     Agent can request any         │
    │     document section by name      │
    │                                   │
    │  3. Few-Shot Examples             │
    │     Show successful extractions   │
    │     from similar documents        │
    │                                   │
    │  4. Search Budget                 │
    │     Max N sections before         │
    │     returning NOT_FOUND           │
    └───────────────────────────────────┘
                    ↓
            Extraction with Citation
```

### Key Features

1. **Chain-of-Thought Reasoning**: LLM explains its search strategy before executing
2. **Few-Shot Examples**: Provide 2-3 successful extraction examples for context
3. **Section Search Tool**: Agent can request specific sections from document map
4. **Search Budget**: Limit iterations to control cost/latency
5. **Learning**: Successful discoveries can update section-to-field mappings

### When to Trigger

- Tier 1 returns `NOT_FOUND` for required field
- Tier 1 confidence is `inferred` with weak evidence quote
- Fallback pattern search returns < 500 chars
- Field has known format variations across fund families

---

## Output Location

Extraction results are saved to `data/extracted/` with naming convention:
- `{fund_name}_extraction.json` - Full extraction result
- `EXTRACTION_GAP_ANALYSIS.md` - Comparison of extraction results

---

*Last updated: January 2025*
