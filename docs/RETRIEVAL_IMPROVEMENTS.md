# Retrieval Improvements for SEC Document Extraction

> **Document Purpose:** This document summarizes the findings from the whole-of-document extraction experiment (January 2026) and outlines structural retrieval improvements needed to scale extraction to 200+ funds.

---

## Executive Summary

After extensive experimentation with the extraction pipeline, **retrieval failures are the primary bottleneck** limiting extraction accuracy. Our error analysis shows:

| Root Cause | % of Errors | Addressable via |
|------------|-------------|-----------------|
| **Retrieval Failure** | 43% | Better retrieval architecture |
| LLM Misinterpretation | 27% | Prompt improvements |
| Ground Truth Nuance | 22% | GT refinement |
| True Hallucination | 8% | Validation |

The current keyword-based retrieval doesn't scale - adding fund-specific keywords for 200+ funds is not viable. This document outlines **structural improvements** that generalize across diverse fund terminology.

---

## Experiment Results: Whole-of-Document Extraction

### Configuration

- **Experiment:** `no_validation_baseline` (Ensemble T4 with no LLM validation)
- **Approach:** Full document chunking with ensemble extraction (T3 keyword + Reranker), T4 agentic escalation for disagreements
- **Validation Set:** 4 funds (Blackstone, StepStone, Hamilton Lane, Blue Owl)
- **Date:** January 2026

### Accuracy Results

| Fund | Accuracy | Correct | Wrong | Missed | Hallucinated |
|------|----------|---------|-------|--------|--------------|
| Blackstone | 94.5% | 52/55 | 3 | 0 | 0 |
| StepStone | 78.0% | 39/50 | 2 | 8 | 1 |
| Hamilton Lane | 86.0% | 43/50 | 2 | 4 | 1 |
| Blue Owl | 68.0% | 34/50 | 4 | 4 | 8 |
| **Overall** | **82.0%** | 168/205 | 11 | 16 | 10 |

### Key Observations

1. **Blackstone performs best** (94.5%) - well-structured document with clear section titles
2. **StepStone has most missed values** (8) - fund-of-funds terminology differs from standard keywords
3. **Blue Owl has most hallucinations** (8) - many relate to ground truth interpretation (intermediary-delegated minimums)
4. **LLM validation doesn't help accuracy** - lightweight validation (82%) = no validation (82%)

---

## Experiment 2: Hybrid Retrieval with Generic Embeddings

### Configuration

- **Experiment:** `hybrid_retrieval_test` (January 2026)
- **Approach:** Hybrid retrieval (keyword + dense embeddings + RRF fusion) replacing keyword-only T3
- **Embedding Model:** `text-embedding-3-small` (OpenAI generic embeddings)
- **Hypothesis:** Hybrid retrieval improves recall by capturing chunks found by either keyword or embedding search
- **Validation Set:** Same 4 funds (Blackstone, StepStone, Hamilton Lane, Blue Owl)

### Results: NEGATIVE

| Fund | Baseline | Hybrid | Change |
|------|----------|--------|--------|
| Blackstone | 94.5% | 81.8% | **-12.7%** |
| StepStone | 78.0% | 60.0% | **-18.0%** |
| Hamilton Lane | 86.0% | 66.0% | **-20.0%** |
| Blue Owl | 68.0% | 52.0% | **-16.0%** |
| **Overall** | **82.0%** | **64.9%** | **-17.1%** |

### Key Observations

1. **Accuracy dropped significantly** across all funds (-17.1% overall)
2. **Agreement rate plummeted** - Only 20.8% of fields agreed between hybrid and reranker methods
3. **Massive T4 escalation** - 79.2% of fields escalated to T4 (vs ~30% in baseline)
4. **T4 couldn't recover** - Even with more T4 calls, accuracy was lower

### Root Cause Analysis

**Generic embeddings don't understand financial terminology:**

The `text-embedding-3-small` model is trained on general web text, not financial documents. It fails to capture domain-specific relationships:

| Query | Expected Match | Actual Behavior |
|-------|----------------|-----------------|
| "hurdle rate" | "preferred return" | Embeddings see these as unrelated |
| "incentive fee" | "performance allocation" | No semantic similarity detected |
| "secondary investments" | "secondaries" | Partial match only |
| "catch-up provision" | "GP catch-up" | Different embedding spaces |

**Consequences:**
- Dense retrieval returns irrelevant chunks
- RRF fusion pollutes good keyword results with bad dense results
- Lower agreement between methods → more T4 escalation
- T4 agent operates on worse context → lower recovery rate

### Lesson Learned

**Generic embeddings hurt rather than help for domain-specific retrieval.** The keyword baseline, while imperfect, at least captures explicit terminology. Generic embeddings introduce noise that overwhelms the signal.

**Next Step:** Test with `voyage-finance-2` (domain-specific financial embeddings trained on SEC filings). This requires a VOYAGE_API_KEY.

---

## Error Classification Deep Dive

### Retrieval Failures (43% of errors)

These are fields where the document contains the information, but retrieval failed to surface relevant chunks to the LLM.

**Examples:**
| Fund | Field | Expected | Issue |
|------|-------|----------|-------|
| StepStone | `secondary_funds_min_pct` | 40 | Document says "40% to 70% in secondary investments" - keywords don't match |
| StepStone | `max_single_fund_pct` | 25 | Document says "no more than 25% in any single Investment Fund" |
| Hamilton Lane | `distribution_fee_pct` | 0.70 | Value in fee table, not prose text |
| Blue Owl | `catch_up_rate_pct` | 100 | Document says "full catch-up" not "100%" |

**Pattern:** Static keyword lists don't cover terminology variations across funds. "Secondary investments", "secondaries", "secondary fund investments" are semantically equivalent but don't match keywords.

### LLM Misinterpretation (27% of errors)

The LLM received correct context but extracted wrong values. Two systematic patterns:

1. **Hurdle rate frequency confusion** (3 occurrences)
   - Document: "5% annualized hurdle rate, paid quarterly in arrears"
   - LLM extracted: "annual" (confused annualized rate with annual frequency)
   - Expected: "quarterly"

2. **Leverage basis/percentage defaults** (4 occurrences)
   - LLM defaults to "33% of total assets" (standard 1940 Act 300% coverage)
   - Misses fund-specific leverage terms like "50% debt-to-equity"

### Ground Truth Interpretation (22% of errors)

The LLM extracted values that **are in the document** but ground truth expects null due to semantic interpretation.

**Example - Blue Owl minimums:**
- Document states: "minimum initial investment... $2,500 for Class S"
- LLM extracted: 2500
- Ground truth: null (note: "delegated to financial intermediaries")

These aren't extraction errors - they're GT definition issues requiring review.

### True Hallucinations (8% of errors)

Genuine cases where LLM invented values not in the document.

**Example - StepStone `has_incentive_fee`:**
- Document: No fund-level incentive fee (underlying funds charge 15-20%)
- LLM extracted: true (confused underlying fund fees with fund-level)

---

## Why Focus on Retrieval?

### 1. Retrieval is the Largest Error Category

43% of all errors stem from retrieval failures. Fixing retrieval has the highest potential impact on overall accuracy.

### 2. Retrieval Errors Cascade

When retrieval fails:
- LLM receives irrelevant context → extracts null or wrong value
- T4 agentic escalation often repeats similar failed searches
- Validation can't catch errors that stem from missing context

### 3. LLM Extraction is Already Good

When retrieval succeeds (correct chunks reach the LLM), extraction accuracy is high. The LLM isn't the bottleneck - getting the right chunks to it is.

### 4. Current Approach Doesn't Scale

Adding keywords per fund is unsustainable for 200+ funds:
- Each fund has unique terminology
- Keyword lists become unmaintainable
- No way to anticipate terminology variations

We need **structural improvements** that generalize, not per-fund configuration.

---

## Current Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Document → Chunking → FIELD_KEYWORDS scoring → Top sections│
│                              │                              │
│                              ├─→ T3: Keyword retrieval      │
│                              │         ↓                    │
│                              │   Cohere Reranker            │
│                              │         ↓                    │
│                              └─→ LLM Extraction             │
│                                        │                    │
│                    If null/disagree → T4 Agentic            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Limitations:**
- Static FIELD_KEYWORDS dictionary doesn't cover terminology variations
- Single query per field - fragile to phrasing
- Hybrid retrieval exists but isn't used in main pipeline
- Dense embeddings use generic model, not financial-specific
- Document structure (section titles) underutilized

---

## Structural Retrieval Improvements

The following improvements are **structural** - they generalize across funds without per-fund configuration.

### 1. Dynamic Query Generation (HyDE)

**Problem:** Static queries miss terminology variations.

**Solution:** Use LLM to generate a hypothetical answer before retrieval. Embed the hypothetical answer instead of the query.

**How it works:**
- Query: "What is the allocation target for secondary investments?"
- LLM generates: "The Fund targets allocations of approximately 40-60% in secondary investments and 20-40% in direct co-investments..."
- Embed this hypothetical answer → matches diverse phrasings

**Why it scales:** The LLM generates domain-appropriate language variations dynamically, without enumeration.

**Research:** HyDE shows 10-20% improvement in zero-shot retrieval (Gao et al., 2022).

---

### 2. Section-Title-First Retrieval

**Problem:** Searching content when SEC N-2 filings have predictable structure.

**Solution:** Use section titles as primary retrieval signal, then search within matched sections.

**SEC N-2 Section Patterns:**
| Field Category | Likely Sections |
|----------------|-----------------|
| Allocation targets | "INVESTMENT OBJECTIVE AND POLICIES", "PRINCIPAL INVESTMENT STRATEGIES" |
| Concentration limits | "INVESTMENT RESTRICTIONS", "FUNDAMENTAL POLICIES" |
| Fee information | "FUND EXPENSES", "MANAGEMENT FEE", "ADVISER COMPENSATION" |
| Repurchase terms | "REPURCHASES AND TRANSFERS OF SHARES", "TENDER OFFERS" |
| Share classes | "PLAN OF DISTRIBUTION", "PURCHASING SHARES" |

**Why it scales:** Section titles are standardized by SEC requirements. "INVESTMENT RESTRICTIONS" means the same thing across all N-2 filings.

**Implementation:** Build section-title → field mapping (one-time effort), use as first-pass filter.

---

### 3. Domain-Specific Embeddings

**Problem:** Generic embeddings don't understand financial terminology relationships.

**Solution:** Switch from `text-embedding-3-small` to `voyage-finance-2`.

**Why it matters:**
- Trained on SEC filings, 10-Ks, financial documents
- Understands: "DRIP" ≈ "dividend reinvestment plan"
- Understands: "secondaries" ≈ "secondary fund investments"

**Already supported** in `embedding_retriever.py` - just needs to be enabled.

**Requirement:** VOYAGE_API_KEY environment variable.

---

### 4. Multi-Query Fusion with RRF

**Problem:** Single query per field is fragile.

**Solution:** Generate 3-5 query variations per field, run all, fuse with Reciprocal Rank Fusion.

**Example for `allocation_targets`:**
1. "What is the target allocation for secondary investments?"
2. "What percentage of assets is invested in secondary funds?"
3. "Investment allocation range for secondaries and co-investments"
4. "Portfolio allocation between direct and secondary investments"

**Fusion:** Chunks found by multiple queries rank highest. RRF formula: `score = Σ 1/(k + rank_i)`

**Why it scales:** Even if one phrasing misses, others catch it. No need to find the "perfect" query.

**Already implemented:** `hybrid_retriever.py` has RRF - extend to multiple query variations.

---

### 5. Enable Hybrid Retrieval as Default

**Problem:** Ensemble runs keyword OR reranker separately. Hybrid retrieval (keyword + dense + RRF) exists but isn't used.

**Solution:** Make hybrid retrieval the default first pass.

**Evidence from our analysis:**
- 15% of correct chunks found by keyword-only
- 10.6% found by dense-only
- Using both captures 25% more correct chunks

**Implementation:** Already built in `hybrid_retriever.py` - wire into ensemble pipeline.

---

### 6. Iterative Retrieval with Confidence Feedback

**Problem:** Single-pass retrieval. If first attempt fails, goes straight to expensive T4 agentic.

**Solution:** Estimate retrieval quality before extraction. If low confidence, reformulate query and retry.

**Approach:**
1. Retrieve top-k chunks
2. Quick relevance check (embedding similarity to query, keyword presence)
3. If low confidence → generate alternative query → retry
4. Only proceed to LLM when retrieval looks adequate

**Why it scales:** Self-corrects retrieval failures before wasting LLM calls.

---

### 7. Chunk Context Enrichment

**Problem:** Chunks embedded in isolation lack context. "40% to 70%" alone doesn't indicate what it refers to.

**Solution:** Prepend section hierarchy to each chunk before embedding.

**Before:**
> "The Fund will target allocations of 40% to 70% in secondary investments."

**After:**
> "[INVESTMENT OBJECTIVE AND POLICIES > Investment Strategies] The Fund will target allocations of 40% to 70% in secondary investments."

**Why it scales:** Section context disambiguates without keywords. Same percentage in "Risk Factors" vs "Investment Strategies" means different things.

**Already supported:** `prepend_context` option in EmbeddingConfig.

---

### 8. Late Interaction Reranking (ColBERT)

**Problem:** Cross-encoder rerankers miss fine-grained token-level matches.

**Solution:** Use ColBERTv2 for reranking (or as supplement to Cohere).

**Why it helps:** ColBERT computes token-level similarity then aggregates. Catches partial matches:
- Query: "secondary investments"
- Document: "secondary fund investments"
- ColBERT sees token overlap that cross-encoders might miss

**Implementation:** RAGatouille library provides easy ColBERT integration.

---

### 9. Table-Aware Retrieval

**Problem:** Many SEC values (minimums, fees) are in tables. Standard chunking breaks table structure.

**Solution:** Detect and handle tables separately:
1. Identify table sections during parsing
2. Extract as structured key-value pairs
3. Index cells with headers as context

**Example:**
```
| Share Class | Minimum Investment | Sales Load |
|-------------|-------------------|------------|
| Class I     | $1,000,000        | 0%         |
```

Becomes searchable as: "Class I minimum investment: $1,000,000"

**Why it scales:** Tables are highly structured - preserving structure makes retrieval trivial.

---

## Implementation Priority

> **Updated after hybrid retrieval experiment (January 2026):** Hybrid retrieval with generic embeddings failed (-17% accuracy). Domain-specific embeddings are now the critical path.

| Priority | Improvement | Effort | Expected Impact | Status |
|----------|-------------|--------|-----------------|--------|
| **1** | **Switch to voyage-finance-2** | Low | +10-20% semantic match | **BLOCKED** - needs VOYAGE_API_KEY |
| **2** | Multi-query with RRF | Medium | +10-15% recall | Query generation needed |
| **3** | Section-title-first retrieval | Medium | +10-15% on structured fields | Section mapping needed |
| **4** | HyDE query expansion | Medium | +10-20% zero-shot | LLM calls |
| **5** | Chunk context enrichment | Low | +5% disambiguation | Config change |
| **6** | ColBERT reranking | Medium | +5-10% precision | New dependency |
| **7** | Table-aware retrieval | High | +15% on table fields | Parser changes |
| **8** | Iterative retrieval | Medium | +5-10% recovery | Confidence scoring |
| ~~9~~ | ~~Enable hybrid retrieval (generic)~~ | ~~Low~~ | ~~-17%~~ | **TESTED - NEGATIVE RESULT** |

**Key Insight:** Hybrid retrieval itself is sound, but requires domain-specific embeddings to work. Generic embeddings add noise that hurts accuracy.

---

## What NOT to Do

| Anti-Pattern | Why It Fails |
|--------------|--------------|
| Add keywords per fund | Doesn't scale to 200+ funds |
| Fine-tune on validation set | Overfits to 4 funds |
| Build per-field rules | Maintenance nightmare |
| Rely on T4 to recover | Expensive, slow, often repeats failures |
| More aggressive validation | Validation can't fix retrieval failures |

---

## Success Metrics

After implementing retrieval improvements, measure:

| Metric | Current | Target |
|--------|---------|--------|
| Overall accuracy | 82% | 90%+ |
| Retrieval failures | 43% of errors | <20% of errors |
| Recall on allocation_targets | ~30% | >80% |
| Recall on concentration_limits | ~30% | >80% |

---

## Next Steps

### Immediate (Blocked on API Key)

1. **Obtain VOYAGE_API_KEY** from [Voyage AI](https://www.voyageai.com/)
2. **Test hybrid retrieval with voyage-finance-2** - Same experiment config, swap embedding model
3. **Validate improvement** - Target: match or exceed 82% baseline

### If voyage-finance-2 Works

4. Add multi-query with RRF for additional recall
5. Implement section-title-first retrieval for structured fields
6. Test on 10 additional funds before full 200+ rollout

### If voyage-finance-2 Fails

4. Fall back to keyword-only T3 (current baseline)
5. Focus on multi-query and HyDE without dense embeddings
6. Consider fine-tuning embeddings on SEC filings corpus

### Experiment Tracking

| Experiment | Date | Result | Accuracy |
|------------|------|--------|----------|
| no_validation_baseline | Jan 2026 | **BASELINE** | 82.0% |
| hybrid_retrieval_test (text-embedding-3-small) | Jan 2026 | NEGATIVE | 64.9% |
| hybrid_retrieval_voyage (voyage-finance-2) | Pending | - | - |

---

## References

- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496) - Gao et al., 2022
- [ColBERTv2: Effective and Efficient Retrieval](https://arxiv.org/abs/2112.01488) - Santhanam et al., 2022
- [Voyage Finance-2 Embeddings](https://docs.voyageai.com/docs/embeddings)
- [FinSage: Multi-aspect RAG for Financial Filings](https://arxiv.org/abs/2504.14493) - April 2025
- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136) - January 2025

---

*Last updated: January 18, 2026 - Added hybrid retrieval experiment results*
