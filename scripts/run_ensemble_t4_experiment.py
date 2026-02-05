#!/usr/bin/env python
"""
Run Ensemble T4 Scenario 2 experiment.

This experiment:
1. Runs T3 k=10 extraction
2. Runs Reranker 200->10 extraction
3. Compares results per field
4. Accepts when both agree on non-null value
5. Escalates to T4 when disagreement OR both null

Expected outcomes (from simulation):
- Accept rate: ~38% of fields with 91% accuracy
- Escalation rate: ~62% (disagreement + both-null)
- T4 success rate: ~80% on escalated fields
- Overall accuracy: ~90%

NOTE ON T4 CHANGES:
- T4 implementation is in pipeline/extract/tier4_agentic.py
- Changes to T4 automatically flow through to this experiment
- No changes needed here unless T4's API signature changes
- If T4 FieldSpec or Tier4Agent API changes, update:
  - pipeline/extract/ensemble_t4_extractor.py (FIELD_PATH_TO_T4_SPEC mapping)
  - pipeline/extract/tier4_agentic.py (FIELD_SPECS dict)
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from pipeline.experiment.config import load_config
from pipeline.experiment.runner import ExperimentRunner
from pipeline.experiment.evaluator import evaluate_run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run Ensemble T4 Scenario 2 experiment."""
    print("=" * 70)
    print("ENSEMBLE T4 SCENARIO 2 EXPERIMENT")
    print("T3 k=10 + Reranker 200->10 + T4 escalation on disagreement/null")
    print("=" * 70)

    # Load config
    config_path = project_root / "configs" / "experiments" / "ensemble_t4_scenario2.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    print(f"\nExperiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"Hypothesis: {config.experiment.hypothesis}")
    print(f"\nExtraction LLM: {config.extraction.provider}/{config.extraction.model}")
    print(f"T4 Model: {config.ensemble.t4_model}")
    print(f"Escalate on disagreement: {config.ensemble.escalate_on_disagreement}")
    print(f"Escalate on both-null: {config.ensemble.escalate_on_both_null}")

    # Verify API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set")
        return 1
    if not os.getenv("COHERE_API_KEY"):
        print("\nWARNING: COHERE_API_KEY not set - reranker may fail")

    # Create runner
    try:
        runner = ExperimentRunner(config)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return 1

    print("\n" + "=" * 70)
    print("RUNNING ENSEMBLE T4 EXTRACTION")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Run T3 k=10 extraction on each fund")
    print("  2. Run Reranker 200->10 extraction on each fund")
    print("  3. Compare results and decide escalations")
    print("  4. Run T4 agentic extraction on escalated fields")
    print("\nEstimated time: 10-20 minutes for 5 funds")
    print("Estimated cost: ~$6 (T3 + Reranker + T4 on ~108 escalated fields)")

    run = runner.run_experiment(name="ensemble_t4_scenario2")

    # Print results summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    summary = run.summarize()
    print(f"\nFunds: {summary['successful_funds']}/{summary['total_funds']}")
    print(f"Duration: {summary['total_duration_seconds']:.1f}s")

    if run.errors:
        print(f"\nErrors:")
        for error in run.errors:
            print(f"  - {error}")

    # Print ensemble-specific stats from traces
    print("\n" + "-" * 70)
    print("ENSEMBLE STATISTICS")
    print("-" * 70)

    total_accepted = 0
    total_escalated = 0
    total_t4_success = 0
    total_fields = 0

    for fund_name, result in run.fund_results.items():
        if result.trace and result.trace.get("extraction_mode") == "ensemble_t4":
            stats = result.trace.get("statistics", {})
            total_accepted += stats.get("accepted_count", 0)
            total_escalated += stats.get("escalated_count", 0)
            total_t4_success += stats.get("t4_success_count", 0)
            total_fields += stats.get("total_fields", 0)

            timing = result.trace.get("timing", {})
            print(f"\n{fund_name}:")
            print(f"  Accepted: {stats.get('accepted_count', 0)}")
            print(f"  Escalated: {stats.get('escalated_count', 0)}")
            print(f"  T4 Successes: {stats.get('t4_success_count', 0)}")
            print(f"  T3 time: {timing.get('t3_duration_seconds', 0):.1f}s")
            print(f"  Reranker time: {timing.get('reranker_duration_seconds', 0):.1f}s")
            print(f"  T4 time: {timing.get('t4_duration_seconds', 0):.1f}s")

    if total_fields > 0:
        print(f"\nAggregate:")
        print(f"  Total fields: {total_fields}")
        print(f"  Accepted: {total_accepted} ({100*total_accepted/total_fields:.1f}%)")
        print(f"  Escalated: {total_escalated} ({100*total_escalated/total_fields:.1f}%)")
        if total_escalated > 0:
            print(f"  T4 Success Rate: {total_t4_success}/{total_escalated} ({100*total_t4_success/total_escalated:.1f}%)")

    # Save results
    output_dir = runner.save_run(run)
    print(f"\nResults saved to: {output_dir}")

    # Evaluate against ground truth
    print("\n" + "=" * 70)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("=" * 70)

    gt_dir = project_root / "configs" / "ground_truth"
    evaluation = evaluate_run(run, str(gt_dir))

    # Save evaluation results
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation.to_dict(), f, indent=2)

    # Print evaluation results
    print(f"\nOverall Accuracy: {evaluation.overall_accuracy:.1%}")
    print(f"Precision: {evaluation.overall_precision:.1%}")
    print(f"Recall: {evaluation.overall_recall:.1%}")
    print(f"F1 Score: {evaluation.overall_f1:.1%}")

    print("\nPer-Fund Results:")
    for fund_name, fund_result in evaluation.fund_results.items():
        print(f"\n  {fund_name}:")
        print(f"    Accuracy: {fund_result.accuracy:.1%}")
        print(f"    Correct: {fund_result.correct_fields}/{fund_result.total_fields}")
        errors = fund_result.error_breakdown()
        print(f"    Errors: wrong={errors['wrong_value']}, missed={errors['missed']}, hallucinated={errors['hallucinated']}")

    # Compare to estimates
    print("\n" + "=" * 70)
    print("COMPARISON TO SIMULATION ESTIMATES")
    print("=" * 70)
    print("\n  Metric          | Estimated | Actual")
    print("  ----------------|-----------|--------")
    if total_fields > 0:
        actual_accept_rate = 100 * total_accepted / total_fields
        actual_escalate_rate = 100 * total_escalated / total_fields
        actual_t4_success = 100 * total_t4_success / total_escalated if total_escalated > 0 else 0
        print(f"  Accept Rate     |   38.3%   | {actual_accept_rate:.1f}%")
        print(f"  Escalate Rate   |   61.7%   | {actual_escalate_rate:.1f}%")
        print(f"  T4 Success Rate |   80.0%   | {actual_t4_success:.1f}%")
    print(f"  Overall Accuracy|   90.3%   | {evaluation.overall_accuracy:.1%}")

    print(f"\nFull evaluation saved to: {eval_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
