# Few-Shot Example Library

This directory contains few-shot examples for LLM extraction. Examples are loaded automatically at runtime.

## File Structure

- `custom_examples.yaml` - **Edit this file** to add your own examples from real filings
- `README.md` - This documentation

## Adding New Examples

### Option 1: Edit YAML directly

Edit `custom_examples.yaml` and add examples in this format:

```yaml
repurchase:  # Category: fees, repurchase, share_classes, allocation, concentration
  - source_text: |
      REPURCHASES OF SHARES

      The Fund will offer quarterly repurchase offers for 5% of outstanding shares...
    extraction:
      fund_structure: interval_fund
      repurchase_frequency: quarterly
      repurchase_percentage_min: 5
      confidence: explicit
    fund_name: "Your Fund Name"
    filing_type: "N-2"
    section_title: "REPURCHASES OF SHARES"
    notes: "Explain why this example is useful (edge cases, etc.)"
    difficulty: "medium"  # easy, medium, or hard
```

### Option 2: Add programmatically

```python
from pipeline.extract.examples import (
    create_example_from_extraction,
    add_example,
    save_examples_to_yaml,
    FieldCategory,
)

# After verifying an extraction is correct:
example = create_example_from_extraction(
    source_text="Your source text here...",
    extraction_result={"field": "value", ...},
    field_category=FieldCategory.REPURCHASE,
    fund_name="StepStone",
    notes="Why this example matters",
)

# Add to runtime library
add_example(example)

# Save all examples to YAML for persistence
save_examples_to_yaml()
```

### Option 3: Export and edit

```python
from pipeline.extract.examples import save_examples_to_yaml

# Export all current examples to YAML
path = save_examples_to_yaml()
print(f"Examples saved to: {path}")

# Edit the file, then reload
from pipeline.extract.examples import reload_examples
reload_examples()
```

## Categories

| Category | Field Names | Description |
|----------|-------------|-------------|
| `fees` | incentive_fee, expense_cap, management_fee | Fee structures |
| `repurchase` | repurchase_terms | Repurchase/tender offer terms |
| `share_classes` | share_classes, minimum_investment | Share class details |
| `allocation` | allocation_targets | Asset allocation targets |
| `concentration` | concentration_limits | Investment concentration limits |

## Best Practices

1. **Use real filing text** - Copy-paste directly from SEC filings for authenticity
2. **Include edge cases** - Examples for "none", "no limit", unusual formats
3. **Add notes** - Explain what makes each example useful
4. **Vary difficulty** - Include easy, medium, and hard examples
5. **Cover both funds** - Examples from StepStone AND Blackstone

## Example Quality Checklist

- [ ] Source text is verbatim from filing
- [ ] Extraction is manually verified as correct
- [ ] Notes explain any non-obvious aspects
- [ ] Difficulty is accurately rated
- [ ] Fund name and section are documented
