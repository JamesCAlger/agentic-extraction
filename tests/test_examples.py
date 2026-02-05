"""
Tests for the few-shot examples module (Phase 2).

Tests cover:
1. ExtractionExample dataclass
2. ExampleLibrary loading and saving
3. Example retrieval by field category
4. Prompt formatting with examples
5. YAML persistence
6. Integration with prompts.py
"""

import pytest
import tempfile
from pathlib import Path

from pipeline.extract.examples import (
    FieldCategory,
    ExtractionExample,
    ExampleLibrary,
    get_examples_for_field,
    format_examples_for_prompt,
    add_example,
    save_examples_to_yaml,
    reload_examples,
    get_example_counts,
    create_example_from_extraction,
    BUILTIN_EXAMPLES,
)
from pipeline.extract.prompts import (
    get_prompt_for_field,
    get_prompt_with_examples,
)


class TestExtractionExample:
    """Tests for ExtractionExample dataclass."""

    def test_creation(self):
        """Test basic example creation."""
        example = ExtractionExample(
            source_text="The Fund offers quarterly repurchases of 5%.",
            extraction={"frequency": "quarterly", "percentage": 5},
            field_category=FieldCategory.REPURCHASE,
        )
        assert example.source_text == "The Fund offers quarterly repurchases of 5%."
        assert example.extraction["frequency"] == "quarterly"
        assert example.field_category == FieldCategory.REPURCHASE

    def test_creation_with_metadata(self):
        """Test example with full metadata."""
        example = ExtractionExample(
            source_text="Test text",
            extraction={"value": 100},
            field_category=FieldCategory.FEES,
            fund_name="StepStone",
            filing_type="N-2",
            section_title="FEES AND EXPENSES",
            notes="Important edge case",
            difficulty="hard",
        )
        assert example.fund_name == "StepStone"
        assert example.filing_type == "N-2"
        assert example.notes == "Important edge case"
        assert example.difficulty == "hard"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        example = ExtractionExample(
            source_text="Test",
            extraction={"key": "value"},
            field_category=FieldCategory.REPURCHASE,
            fund_name="Test Fund",
        )
        d = example.to_dict()

        assert d["source_text"] == "Test"
        assert d["extraction"] == {"key": "value"}
        assert d["field_category"] == "repurchase"  # String, not enum
        assert d["fund_name"] == "Test Fund"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "source_text": "Test text",
            "extraction": {"value": 123},
            "field_category": "fees",
            "fund_name": "Test Fund",
            "difficulty": "easy",
        }
        example = ExtractionExample.from_dict(d)

        assert example.source_text == "Test text"
        assert example.field_category == FieldCategory.FEES
        assert example.difficulty == "easy"

    def test_format_for_prompt(self):
        """Test formatting example for prompt."""
        example = ExtractionExample(
            source_text="The Fund charges a 1.5% management fee.",
            extraction={"management_fee": 1.5, "confidence": "explicit"},
            field_category=FieldCategory.FEES,
            notes="Standard fee extraction",
        )
        formatted = example.format_for_prompt()

        assert "--- EXAMPLE ---" in formatted
        assert "The Fund charges a 1.5% management fee." in formatted
        assert '"management_fee": 1.5' in formatted
        assert "Standard fee extraction" in formatted


class TestExampleLibrary:
    """Tests for ExampleLibrary class."""

    def test_empty_library(self):
        """Test empty library creation."""
        library = ExampleLibrary()
        assert len(library.examples) == 0

    def test_add_example(self):
        """Test adding examples."""
        library = ExampleLibrary()
        example = ExtractionExample(
            source_text="Test",
            extraction={},
            field_category=FieldCategory.REPURCHASE,
        )
        library.add_example(example)

        assert FieldCategory.REPURCHASE in library.examples
        assert len(library.examples[FieldCategory.REPURCHASE]) == 1

    def test_get_examples(self):
        """Test retrieving examples."""
        library = ExampleLibrary()

        # Add multiple examples
        for i in range(5):
            library.add_example(ExtractionExample(
                source_text=f"Example {i}",
                extraction={"id": i},
                field_category=FieldCategory.FEES,
            ))

        # Get limited examples
        examples = library.get_examples(FieldCategory.FEES, max_examples=3)
        assert len(examples) == 3

    def test_get_examples_by_difficulty(self):
        """Test filtering by difficulty."""
        library = ExampleLibrary()

        library.add_example(ExtractionExample(
            source_text="Easy", extraction={},
            field_category=FieldCategory.FEES, difficulty="easy",
        ))
        library.add_example(ExtractionExample(
            source_text="Hard", extraction={},
            field_category=FieldCategory.FEES, difficulty="hard",
        ))

        easy_examples = library.get_examples(FieldCategory.FEES, difficulty="easy")
        assert len(easy_examples) == 1
        assert easy_examples[0].source_text == "Easy"

    def test_save_and_load_yaml(self):
        """Test YAML persistence."""
        library = ExampleLibrary()
        library.add_example(ExtractionExample(
            source_text="Quarterly repurchases of 5%",
            extraction={"frequency": "quarterly", "percentage": 5},
            field_category=FieldCategory.REPURCHASE,
            fund_name="Test Fund",
            notes="Test note",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_examples.yaml"

            # Save
            library.save_to_yaml(yaml_path)
            assert yaml_path.exists()

            # Load
            loaded = ExampleLibrary.load_from_yaml(yaml_path)

            assert FieldCategory.REPURCHASE in loaded.examples
            assert len(loaded.examples[FieldCategory.REPURCHASE]) == 1
            loaded_ex = loaded.examples[FieldCategory.REPURCHASE][0]
            assert loaded_ex.source_text == "Quarterly repurchases of 5%"
            assert loaded_ex.fund_name == "Test Fund"

    def test_count_examples(self):
        """Test counting examples by category."""
        library = ExampleLibrary()

        library.add_example(ExtractionExample(
            source_text="A", extraction={}, field_category=FieldCategory.FEES))
        library.add_example(ExtractionExample(
            source_text="B", extraction={}, field_category=FieldCategory.FEES))
        library.add_example(ExtractionExample(
            source_text="C", extraction={}, field_category=FieldCategory.REPURCHASE))

        counts = library.count_examples()
        assert counts["fees"] == 2
        assert counts["repurchase"] == 1


class TestBuiltinExamples:
    """Tests for builtin examples."""

    def test_builtin_examples_exist(self):
        """Test that builtin examples are defined."""
        assert len(BUILTIN_EXAMPLES) > 0

    def test_builtin_examples_cover_categories(self):
        """Test that builtins cover main categories."""
        categories = {ex.field_category for ex in BUILTIN_EXAMPLES}

        assert FieldCategory.REPURCHASE in categories
        assert FieldCategory.SHARE_CLASSES in categories
        assert FieldCategory.ALLOCATION in categories
        assert FieldCategory.CONCENTRATION in categories
        assert FieldCategory.FEES in categories

    def test_builtin_repurchase_examples(self):
        """Test repurchase examples quality."""
        repurchase_examples = [
            ex for ex in BUILTIN_EXAMPLES
            if ex.field_category == FieldCategory.REPURCHASE
        ]
        assert len(repurchase_examples) >= 2

        # Check they have required content
        for ex in repurchase_examples:
            assert len(ex.source_text) > 100  # Realistic text length
            assert ex.extraction  # Non-empty extraction
            assert "repurchase" in ex.extraction or "fund_structure" in ex.extraction

    def test_builtin_share_class_examples(self):
        """Test share class examples."""
        share_examples = [
            ex for ex in BUILTIN_EXAMPLES
            if ex.field_category == FieldCategory.SHARE_CLASSES
        ]
        assert len(share_examples) >= 2

        # Should have share_classes in extraction
        for ex in share_examples:
            assert "share_classes" in ex.extraction


class TestGetExamplesForField:
    """Tests for get_examples_for_field function."""

    def test_get_repurchase_examples(self):
        """Test getting repurchase examples."""
        examples = get_examples_for_field("repurchase_terms")
        assert len(examples) > 0
        assert all(ex.field_category == FieldCategory.REPURCHASE for ex in examples)

    def test_get_share_class_examples(self):
        """Test getting share class examples."""
        examples = get_examples_for_field("share_classes")
        assert len(examples) > 0
        assert all(ex.field_category == FieldCategory.SHARE_CLASSES for ex in examples)

    def test_get_allocation_examples(self):
        """Test getting allocation examples."""
        examples = get_examples_for_field("allocation_targets")
        assert len(examples) > 0
        assert all(ex.field_category == FieldCategory.ALLOCATION for ex in examples)

    def test_get_fee_examples(self):
        """Test getting fee examples."""
        examples = get_examples_for_field("incentive_fee")
        assert len(examples) > 0

    def test_max_examples_limit(self):
        """Test max_examples parameter."""
        examples = get_examples_for_field("repurchase_terms", max_examples=1)
        assert len(examples) <= 1

    def test_unknown_field(self):
        """Test with unknown field name."""
        examples = get_examples_for_field("unknown_field_xyz")
        assert examples == []


class TestFormatExamplesForPrompt:
    """Tests for format_examples_for_prompt function."""

    def test_format_empty(self):
        """Test formatting empty list."""
        result = format_examples_for_prompt([])
        assert result == ""

    def test_format_single_example(self):
        """Test formatting single example."""
        example = ExtractionExample(
            source_text="Test source text",
            extraction={"key": "value"},
            field_category=FieldCategory.FEES,
            notes="Test note",
        )
        result = format_examples_for_prompt([example])

        assert "## Examples" in result
        assert "Example 1" in result
        assert "Test source text" in result
        assert '"key": "value"' in result
        assert "Test note" in result

    def test_format_without_notes(self):
        """Test formatting without notes."""
        example = ExtractionExample(
            source_text="Test",
            extraction={"x": 1},
            field_category=FieldCategory.FEES,
            notes="Should not appear",
        )
        result = format_examples_for_prompt([example], include_notes=False)

        assert "Should not appear" not in result

    def test_format_multiple_examples(self):
        """Test formatting multiple examples."""
        examples = [
            ExtractionExample(
                source_text=f"Example {i}",
                extraction={"id": i},
                field_category=FieldCategory.REPURCHASE,
            )
            for i in range(3)
        ]
        result = format_examples_for_prompt(examples)

        assert "Example 1" in result
        assert "Example 2" in result
        assert "Example 3" in result


class TestGetPromptWithExamples:
    """Tests for get_prompt_with_examples function."""

    def test_prompt_includes_examples(self):
        """Test that examples are included in prompt."""
        prompt = get_prompt_with_examples("repurchase_terms", max_examples=2)

        assert "## Examples" in prompt
        assert "Extract share repurchase" in prompt  # Base prompt content
        assert "Now extract from this text:" in prompt

    def test_prompt_preserves_text_placeholder(self):
        """Test that {text} placeholder is preserved."""
        prompt = get_prompt_with_examples("share_classes")

        assert "{text}" in prompt

    def test_unknown_field_returns_empty(self):
        """Test unknown field returns empty string."""
        prompt = get_prompt_with_examples("unknown_field")
        assert prompt == ""

    def test_base_prompt_still_works(self):
        """Test get_prompt_for_field still works without examples."""
        prompt = get_prompt_for_field("repurchase_terms")

        assert "Extract share repurchase" in prompt
        assert "## Examples" not in prompt


class TestCreateExampleFromExtraction:
    """Tests for create_example_from_extraction helper."""

    def test_create_example(self):
        """Test creating example from extraction result."""
        example = create_example_from_extraction(
            source_text="The Fund repurchases shares quarterly.",
            extraction_result={"frequency": "quarterly"},
            field_category=FieldCategory.REPURCHASE,
            fund_name="Test Fund",
            notes="Basic quarterly example",
        )

        assert example.source_text == "The Fund repurchases shares quarterly."
        assert example.extraction == {"frequency": "quarterly"}
        assert example.field_category == FieldCategory.REPURCHASE
        assert example.fund_name == "Test Fund"


class TestExampleCounts:
    """Tests for get_example_counts function."""

    def test_counts_include_builtins(self):
        """Test that counts include builtin examples."""
        counts = get_example_counts()

        # Should have examples in multiple categories
        assert sum(counts.values()) > 0
        assert "repurchase" in counts
        assert "fees" in counts


class TestSaveExamplesToYaml:
    """Tests for save_examples_to_yaml function."""

    def test_save_to_custom_path(self):
        """Test saving to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_examples.yaml"

            result_path = save_examples_to_yaml(output_path)

            assert result_path == output_path
            assert output_path.exists()

            # Verify content
            import yaml
            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert data is not None
            assert any(len(examples) > 0 for examples in data.values())


class TestAddExampleAndReload:
    """Tests for add_example and reload_examples functions."""

    def test_add_example(self):
        """Test adding example to global library."""
        initial_counts = get_example_counts()

        # Add a new example
        add_example(ExtractionExample(
            source_text="New test example",
            extraction={"test": True},
            field_category=FieldCategory.CONCENTRATION,
        ))

        new_counts = get_example_counts()

        assert new_counts.get("concentration", 0) >= initial_counts.get("concentration", 0)

        # Reload to reset
        reload_examples()


class TestIntegration:
    """Integration tests for the examples module."""

    def test_full_workflow(self):
        """Test complete workflow: create, format, use in prompt."""
        # 1. Create an example
        example = create_example_from_extraction(
            source_text="""REPURCHASES OF SHARES

The Fund is an interval fund that offers to repurchase 5% of outstanding
shares on a quarterly basis.""",
            extraction_result={
                "fund_structure": "interval_fund",
                "repurchase_frequency": "quarterly",
                "repurchase_percentage": 5,
                "confidence": "explicit",
            },
            field_category=FieldCategory.REPURCHASE,
            fund_name="Test Fund",
            notes="Standard interval fund",
        )

        # 2. Format for prompt
        formatted = format_examples_for_prompt([example])

        assert "interval fund" in formatted.lower()
        assert "quarterly" in formatted

        # 3. Get prompt with examples
        prompt = get_prompt_with_examples("repurchase_terms")

        assert "## Examples" in prompt
        assert "{text}" in prompt

    def test_examples_loaded_from_multiple_sources(self):
        """Test that examples come from builtins."""
        # Reload to ensure clean state
        reload_examples()

        # Should have examples from builtins
        counts = get_example_counts()
        total = sum(counts.values())

        assert total >= len(BUILTIN_EXAMPLES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
