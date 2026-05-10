"""
Plot cases where showing CoT reasoning makes the judge perform worse.

This script identifies datapoints where:
- Without CoT (constitution_and_response): Judge correctly catches violation
- With CoT (constitution_and_reasoning_and_response): Judge fails to catch violation

These are cases where showing the model's reasoning actually degrades judge performance.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple
import seaborn as sns

# Set style for publication-ready plots (matching plot_reward.py)
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.linewidth': 0.8,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.8,
    'lines.markersize': 4.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2
})

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Force serif font after seaborn style
import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']


def load_datapoint_ids(jsonl_path: Path) -> Set[int]:
    """Load example_index values from a category JSONL file."""
    ids = set()

    if not jsonl_path.exists():
        return ids

    with open(jsonl_path, 'r') as f:
        content = f.read()

    # Handle pretty-printed JSON objects separated by newlines
    # Split on "}\n{" pattern and reconstruct objects
    objects = []
    current_obj = ""

    for line in content.split('\n'):
        current_obj += line + '\n'
        if line.strip() == '}':
            # End of an object
            try:
                data = json.loads(current_obj)
                ids.add(data['example_index'])
                current_obj = ""
            except (json.JSONDecodeError, KeyError):
                # Continue accumulating
                pass

    return ids


def load_summary(summary_path: Path) -> Dict:
    """Load summary.json file."""
    if not summary_path.exists():
        return {}

    with open(summary_path, 'r') as f:
        return json.load(f)


def compute_cot_degradation(base_path: Path, evaluator: str,
                           reasonableness_version: str) -> Dict[int, Dict]:
    """
    Compute metrics for cases where CoT makes judge worse.

    Args:
        base_path: Path to experiment analysis directory (e.g., analysis_output/.../now.../later_constitutional_cot_v2)
        evaluator: Evaluator name
        reasonableness_version: Version of reasonableness prompt

    Returns:
        Dict mapping iteration -> metrics dict
    """
    # Paths to eval targets
    without_cot_base = (base_path / "constitution_and_response" /
                        f"evaluator-{evaluator}" / reasonableness_version)
    with_cot_base = (base_path / "constitution_and_reasoning_and_response" /
                     f"evaluator-{evaluator}" / reasonableness_version)

    if not without_cot_base.exists() or not with_cot_base.exists():
        print(f"Warning: Missing evaluation directories for {base_path.name}")
        return {}

    results = {}

    # Find all iterations
    iterations = set()
    for iteration_dir in without_cot_base.iterdir():
        if iteration_dir.is_dir() and iteration_dir.name.startswith("iteration-"):
            iteration_str = iteration_dir.name.replace("iteration-", "")
            if iteration_str != "base":
                try:
                    iterations.add(int(iteration_str))
                except ValueError:
                    continue

    for iteration in sorted(iterations):
        # Paths for this iteration
        without_cot_caught = without_cot_base / f"iteration-{iteration}" / "incorrect_unreasonable.jsonl"
        with_cot_missed = with_cot_base / f"iteration-{iteration}" / "incorrect_reasonable.jsonl"
        with_cot_summary = with_cot_base / f"iteration-{iteration}" / "summary.json"

        # Load datapoint IDs
        caught_without_cot = load_datapoint_ids(without_cot_caught)
        missed_with_cot = load_datapoint_ids(with_cot_missed)

        # Find intersection: caught without CoT but missed with CoT
        cot_degradation_ids = caught_without_cot & missed_with_cot
        cot_degradation_count = len(cot_degradation_ids)

        # Total caught without CoT (denominator for rate)
        total_caught_without_cot = len(caught_without_cot)

        # Load summary for totals
        summary = load_summary(with_cot_summary)
        if not summary:
            continue

        category_counts = summary['statistics']['category_counts']
        total_processed = summary['statistics']['total_processed']

        # Total incorrect in with-CoT evaluation
        total_incorrect_with_cot = (
            category_counts.get('incorrect_reasonable', 0) +
            category_counts.get('incorrect_unreasonable', 0)
        )

        # Compute metrics
        # Degradation rate: Of cases caught without CoT, what % were missed with CoT?
        # This tells us how much worse CoT makes things
        if total_caught_without_cot > 0:
            degradation_rate = cot_degradation_count / total_caught_without_cot
        else:
            degradation_rate = 0.0

        if total_processed > 0:
            degradation_proportion = cot_degradation_count / total_processed
        else:
            degradation_proportion = 0.0

        results[iteration] = {
            'cot_degradation_count': cot_degradation_count,
            'total_caught_without_cot': total_caught_without_cot,
            'total_incorrect_with_cot': total_incorrect_with_cot,
            'total_processed': total_processed,
            'cot_degradation_rate': degradation_rate,
            'cot_degradation_proportion': degradation_proportion
        }

    return results


def plot_single_metric(experiments_data: List[Dict], output_path: Path,
                      metric_key: str, title: str, ylabel: str, use_percentage: bool = True):
    """Create a single line plot for a specific metric."""
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Force serif font
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Colors matching plot_reward.py order: Risky, Safe, Now, Later
    colors = ['#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A']

    max_value = 0
    for idx, exp_data in enumerate(experiments_data):
        experiment_name = exp_data['experiment_name']
        metrics = exp_data['metrics']

        if not metrics:
            continue

        # Sort iterations and limit to first 9 (0-8)
        iterations = sorted([it for it in metrics.keys() if it <= 8])
        values = [metrics[it][metric_key] for it in iterations]

        # Track max value for y-axis limits when showing counts
        if not use_percentage:
            max_value = max(max_value, max(values))

        color = colors[idx % len(colors)]
        ax.plot(iterations, values, marker='o', linewidth=2, markersize=6,
               color=color, alpha=0.7, label=experiment_name)

    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)

    if use_percentage:
        ax.set_ylim(0, 1)
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    else:
        # For counts, set y-axis limit with some headroom
        ax.set_ylim(0, max_value * 1.1)
        # Format as integers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    plt.tight_layout()

    # Save plots
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved plot to: {output_path}")

    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved PDF to: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot CoT degradation metrics (cases where showing CoT makes judge worse)'
    )
    parser.add_argument('--all-experiments', action='store_true',
                       help='Plot all four experiments')
    parser.add_argument('--output-base-dir', default='plots/cot_degradation',
                       help='Base output directory for plots')
    parser.add_argument('--evaluator', default='3-8b-instruct',
                       help='Evaluator name')
    parser.add_argument('--reasonableness-version', default='simple_reasonable_recommendation_v3',
                       help='Reasonableness version')

    args = parser.parse_args()

    # Define experiments (matching plot_reward.py order)
    if args.all_experiments:
        analysis_dirs = [
            ('analysis_output/reasonableness/risky-09_19_182001/safe_constitutional_cot_v2', 'Risky'),
            ('analysis_output/reasonableness/safe-09_19_182118/risky_constitutional_cot_v2', 'Safe'),
            ('analysis_output/reasonableness/now-09_20_201429/later_constitutional_cot_v2', 'Now'),
            ('analysis_output/reasonableness/later-09_20_201440/now_constitutional_cot_v2', 'Later'),
        ]
    else:
        print("Error: Must specify --all-experiments")
        return

    print(f"Evaluator: {args.evaluator}")
    print(f"Reasonableness version: {args.reasonableness_version}")
    print()

    # Compute metrics for all experiments
    experiments_data = []

    for analysis_dir, experiment_name in analysis_dirs:
        analysis_path = Path(analysis_dir)

        if not analysis_path.exists():
            print(f"Warning: Analysis directory does not exist: {analysis_path}")
            continue

        print(f"Computing CoT degradation metrics for {experiment_name}...")
        metrics = compute_cot_degradation(analysis_path, args.evaluator, args.reasonableness_version)

        if metrics:
            experiments_data.append({
                'experiment_name': experiment_name,
                'metrics': metrics
            })
            print(f"  Loaded {len(metrics)} iterations")

            # Print some stats
            total_degradation = sum(m['cot_degradation_count'] for m in metrics.values())
            print(f"  Total CoT degradation cases: {total_degradation}")
        else:
            print(f"  No data found")
        print()

    if not experiments_data:
        print("Error: No data found for any experiment!")
        return

    # Create output path
    output_base = Path(args.output_base_dir) / f"evaluator-{args.evaluator}" / args.reasonableness_version

    # Plot 1: CoT Degradation Rate (of cases caught without CoT, what % missed with CoT)
    rate_path = output_base / "cot_degradation_rate.png"
    plot_single_metric(
        experiments_data,
        rate_path,
        'cot_degradation_rate',
        'CoT Monitor Degradation Rate',
        'Fraction Caught Without CoT,\nMissed With CoT',
        use_percentage=True
    )

    # Plot 2: CoT Degradation Count (raw numbers instead of proportion)
    count_path = output_base / "cot_degradation_count.png"
    plot_single_metric(
        experiments_data,
        count_path,
        'cot_degradation_count',
        'Monitor Tricked by CoT',
        'Number of Datapoints',
        use_percentage=False
    )

    print(f"\n{'='*80}")
    print("PLOTTING COMPLETE!")
    print(f"Plots saved to: {output_base}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
