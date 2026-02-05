"""
CLI interface for extraction experiments.

Usage:
    python -m pipeline.experiment run --config configs/base.yaml
    python -m pipeline.experiment run --config configs/experiments/per_section.yaml --name my_test
    python -m pipeline.experiment evaluate --run exp_20260109_baseline
    python -m pipeline.experiment compare --baseline exp_1 --variant exp_2
    python -m pipeline.experiment list
    python -m pipeline.experiment trace --run exp_1 --fund "Blackstone" --field "incentive_fee"
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_run(args):
    """Run an extraction experiment."""
    from .config import load_config, validate_config
    from .runner import ExperimentRunner

    # Load config
    config = load_config(args.config)

    # Validate
    warnings = validate_config(config)
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")

    # Print experiment info
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Name: {args.name or config.experiment.name or 'baseline'}")
    print(f"Config hash: {config.config_hash()}")
    print(f"Extraction mode: {config.extraction.extraction_mode}")
    print(f"Model: {config.extraction.model}")
    print(f"Tiers enabled: T0={config.tiers.tier0_enabled}, T1={config.tiers.tier1_enabled}, "
          f"T2={config.tiers.tier2_enabled}, T3={config.tiers.tier3_enabled}")
    print(f"Grounding: {config.grounding.enabled}")
    print(f"Validation funds: {config.validation.funds or 'all defaults'}")
    print()

    # Create runner
    runner = ExperimentRunner(config, output_dir=args.output_dir)

    # Run experiment
    run = runner.run_experiment(name=args.name)

    # Save results
    exp_dir = runner.save_run(run)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {run.run_id}")
    print(f"Output: {exp_dir}")

    summary = run.summarize()
    print(f"Funds: {summary['successful_funds']}/{summary['total_funds']} successful")
    print(f"Duration: {summary['total_duration_seconds']:.1f}s")

    if summary['errors']:
        print(f"\nErrors:")
        for error in summary['errors']:
            print(f"  - {error}")

    # Auto-evaluate if ground truth available
    if args.evaluate:
        print(f"\n{'='*60}")
        print(f"EVALUATING AGAINST GROUND TRUTH")
        print(f"{'='*60}")
        cmd_evaluate_internal(run, config.validation.ground_truth_dir)

    return run


def cmd_evaluate(args):
    """Evaluate an experiment run against ground truth."""
    from .runner import load_run
    from .config import load_config

    # Load run
    run_path = Path(args.output_dir) / args.run
    if not run_path.exists():
        # Try as full path
        run_path = Path(args.run)

    if not run_path.exists():
        print(f"Error: Run not found: {args.run}")
        sys.exit(1)

    run = load_run(run_path)

    # Get ground truth dir
    gt_dir = args.ground_truth or run.config.validation.ground_truth_dir

    cmd_evaluate_internal(run, gt_dir, args.output)


def cmd_evaluate_internal(run, gt_dir: str, output_path: str = None):
    """Internal evaluate function used by both run and evaluate commands."""
    from .evaluator import evaluate_run, print_evaluation_report

    evaluation = evaluate_run(run, gt_dir)

    # Print report
    report = print_evaluation_report(
        evaluation,
        show_field_details=True,
    )
    print(report)

    # Save evaluation results
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = Path(f"data/experiments/{run.run_id}/evaluation.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation.to_dict(), f, indent=2, default=str)

    print(f"\nEvaluation saved to: {output_path}")


def cmd_compare(args):
    """Compare two experiment runs."""
    from .runner import load_run
    from .evaluator import evaluate_run, EvaluationResult

    # Load both runs
    baseline_path = Path(args.output_dir) / args.baseline
    variant_path = Path(args.output_dir) / args.variant

    if not baseline_path.exists():
        baseline_path = Path(args.baseline)
    if not variant_path.exists():
        variant_path = Path(args.variant)

    baseline_run = load_run(baseline_path)
    variant_run = load_run(variant_path)

    # Get ground truth dir
    gt_dir = args.ground_truth or baseline_run.config.validation.ground_truth_dir

    # Evaluate both
    baseline_eval = evaluate_run(baseline_run, gt_dir)
    variant_eval = evaluate_run(variant_run, gt_dir)

    # Generate comparison report
    report = generate_comparison_report(
        baseline_run,
        variant_run,
        baseline_eval,
        variant_eval,
    )

    print(report)

    # Save report
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


def generate_comparison_report(
    baseline_run,
    variant_run,
    baseline_eval,
    variant_eval,
) -> str:
    """Generate a comparison report between two experiments."""
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("EXTRACTION EXPERIMENT COMPARISON")
    lines.append(sep)
    lines.append(
        f"Baseline: {baseline_run.run_id} "
        f"({baseline_run.config.extraction.extraction_mode}, "
        f"{baseline_run.config.extraction.max_chunk_tokens} tokens)"
    )
    lines.append(
        f"Variant:  {variant_run.run_id} "
        f"({variant_run.config.extraction.extraction_mode}, "
        f"{variant_run.config.extraction.max_chunk_tokens} tokens)"
    )
    lines.append("")

    # Overall metrics comparison
    lines.append("OVERALL METRICS")
    lines.append(f"{'Metric':<20} {'Baseline':>12} {'Variant':>12} {'Delta':>12}")
    lines.append("-" * 60)

    metrics = [
        ("Precision", baseline_eval.overall_precision, variant_eval.overall_precision),
        ("Recall", baseline_eval.overall_recall, variant_eval.overall_recall),
        ("F1 Score", baseline_eval.overall_f1, variant_eval.overall_f1),
        ("Accuracy", baseline_eval.overall_accuracy, variant_eval.overall_accuracy),
    ]

    for name, baseline_val, variant_val in metrics:
        delta = variant_val - baseline_val
        delta_str = f"{delta:+.3f}" if delta != 0 else "0"
        lines.append(
            f"{name:<20} {baseline_val:>12.3f} {variant_val:>12.3f} {delta_str:>12}"
        )
    lines.append("")

    # Per-fund comparison
    lines.append("PER-FUND BREAKDOWN")
    lines.append(f"{'Fund':<30} {'Baseline':>12} {'Variant':>12} {'Change':>12}")
    lines.append("-" * 60)

    all_funds = set(baseline_eval.fund_results.keys()) | set(variant_eval.fund_results.keys())
    for fund_name in sorted(all_funds):
        baseline_fund = baseline_eval.fund_results.get(fund_name)
        variant_fund = variant_eval.fund_results.get(fund_name)

        baseline_score = f"{baseline_fund.correct_fields}/{baseline_fund.total_fields}" if baseline_fund else "N/A"
        variant_score = f"{variant_fund.correct_fields}/{variant_fund.total_fields}" if variant_fund else "N/A"

        if baseline_fund and variant_fund:
            diff = variant_fund.correct_fields - baseline_fund.correct_fields
            change = f"{diff:+d} fields" if diff != 0 else "="
        else:
            change = "N/A"

        short_name = fund_name[:28] if len(fund_name) > 28 else fund_name
        lines.append(f"{short_name:<30} {baseline_score:>12} {variant_score:>12} {change:>12}")
    lines.append("")

    # Field-level changes
    lines.append("FIELD-LEVEL CHANGES")
    lines.append("-" * 60)

    improved = []
    regressed = []

    baseline_fields = baseline_eval._per_field_accuracy()
    variant_fields = variant_eval._per_field_accuracy()

    all_fields = set(baseline_fields.keys()) | set(variant_fields.keys())
    for field_path in all_fields:
        baseline_acc = baseline_fields.get(field_path, {}).get("accuracy", 0)
        variant_acc = variant_fields.get(field_path, {}).get("accuracy", 0)

        if variant_acc > baseline_acc:
            improved.append((field_path, baseline_acc, variant_acc))
        elif variant_acc < baseline_acc:
            regressed.append((field_path, baseline_acc, variant_acc))

    if improved:
        lines.append("+ IMPROVED (Variant better than Baseline):")
        for field_path, base_acc, var_acc in improved[:10]:
            lines.append(f"    {field_path}: {base_acc:.0%} -> {var_acc:.0%}")
    else:
        lines.append("+ IMPROVED: None")
    lines.append("")

    if regressed:
        lines.append("- REGRESSED (Variant worse than Baseline):")
        for field_path, base_acc, var_acc in regressed[:10]:
            lines.append(f"    {field_path}: {base_acc:.0%} -> {var_acc:.0%}")
    else:
        lines.append("- REGRESSED: None")
    lines.append("")

    # Recommendation
    lines.append("RECOMMENDATION")
    lines.append("-" * 60)
    if variant_eval.overall_f1 > baseline_eval.overall_f1:
        lines.append("Variant OUTPERFORMS baseline. Consider adopting.")
    elif variant_eval.overall_f1 < baseline_eval.overall_f1:
        lines.append("Baseline outperforms variant. Do NOT adopt.")
    else:
        lines.append("No significant difference between baseline and variant.")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def cmd_list(args):
    """List all experiment runs."""
    from .runner import list_runs

    runs = list_runs(args.output_dir)

    if not runs:
        print("No experiment runs found.")
        return

    print(f"\n{'='*80}")
    print(f"EXPERIMENT RUNS")
    print(f"{'='*80}")
    print(f"{'Run ID':<40} {'Funds':>8} {'Started':<20}")
    print("-" * 80)

    for run in runs:
        funds = f"{run['successful_funds']}/{run['total_funds']}"
        started = run['started_at'][:19] if run['started_at'] else "N/A"
        print(f"{run['run_id']:<40} {funds:>8} {started:<20}")

    print(f"\nTotal: {len(runs)} runs in {args.output_dir}")


def cmd_trace(args):
    """Show detailed trace for a specific field."""
    from .runner import load_run

    # Load run
    run_path = Path(args.output_dir) / args.run
    if not run_path.exists():
        run_path = Path(args.run)

    run = load_run(run_path)

    # Find fund
    fund_result = None
    for name, result in run.fund_results.items():
        if args.fund.lower() in name.lower():
            fund_result = result
            break

    if not fund_result:
        print(f"Fund not found: {args.fund}")
        print(f"Available funds: {list(run.fund_results.keys())}")
        sys.exit(1)

    # Get trace
    if not fund_result.trace:
        print(f"No trace available for {fund_result.fund_name}")
        sys.exit(1)

    trace = fund_result.trace

    if args.field:
        # Show specific field
        extractions = trace.get("extractions", {})
        if args.field in extractions:
            print(json.dumps(extractions[args.field], indent=2, default=str))
        else:
            print(f"Field not found: {args.field}")
            print(f"Available fields: {list(extractions.keys())}")
    else:
        # Show summary
        print(f"\n{'='*60}")
        print(f"TRACE: {fund_result.fund_name}")
        print(f"{'='*60}")

        summary = trace.get("summary", {})
        print(f"Total fields: {summary.get('total_fields', 'N/A')}")
        print(f"Flagged for review: {summary.get('flagged_for_review', 'N/A')}")
        print(f"Duration: {summary.get('duration_seconds', 'N/A')}s")

        print("\nBy confidence:")
        for level, count in summary.get("by_confidence", {}).items():
            print(f"  {level}: {count}")

        print("\nBy source layer:")
        for layer, count in summary.get("by_source_layer", {}).items():
            print(f"  {layer}: {count}")

        print("\nAvailable fields for --field:")
        for field_name in trace.get("extractions", {}).keys():
            print(f"  {field_name}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extraction Experimentation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        default="data/experiments",
        help="Base directory for experiment outputs (default: data/experiments)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an extraction experiment")
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (base.yaml or experiment override)",
    )
    run_parser.add_argument(
        "--name",
        help="Experiment name (overrides config)",
    )
    run_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Automatically evaluate against ground truth after run",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate against ground truth")
    eval_parser.add_argument(
        "--run",
        required=True,
        help="Run ID or path to evaluate",
    )
    eval_parser.add_argument(
        "--ground-truth",
        help="Path to ground truth directory (default: from config)",
    )
    eval_parser.add_argument(
        "--output",
        help="Path to save evaluation results",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two experiment runs")
    compare_parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline run ID or path",
    )
    compare_parser.add_argument(
        "--variant",
        required=True,
        help="Variant run ID or path",
    )
    compare_parser.add_argument(
        "--ground-truth",
        help="Path to ground truth directory",
    )
    compare_parser.add_argument(
        "--output",
        help="Path to save comparison report",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all experiment runs")

    # Trace command
    trace_parser = subparsers.add_parser("trace", help="Show extraction trace details")
    trace_parser.add_argument(
        "--run",
        required=True,
        help="Run ID or path",
    )
    trace_parser.add_argument(
        "--fund",
        required=True,
        help="Fund name (partial match)",
    )
    trace_parser.add_argument(
        "--field",
        help="Specific field to show (omit for summary)",
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "trace":
        cmd_trace(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
