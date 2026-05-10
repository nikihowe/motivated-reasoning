"""
Plot reasonableness category proportions over training iterations.

This script creates line plots showing how the proportion of different
correctness/reasonableness categories change over training iterations.

Usage:
    python plot_reasonableness_proportions.py analysis_dir [--output-dir OUTPUT_DIR]

Example:
    python plot_reasonableness_proportions.py analysis_output/reasonableness/now-09_20_201429/later_constitutional_cot_v2
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
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

# Style matching plot_reward.py
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Force serif font after seaborn style (which may override font settings)
import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']


def load_iteration_data(base_path: Path, eval_target: str, evaluator: str, reasonableness_version: str = None) -> Dict[int, Dict]:
    """
    Load summary data for all iterations of a specific eval_target.

    Args:
        base_path: Base path to the analysis directory (should be at prompt_type level)
        eval_target: Eval target name (e.g., constitution_and_response)
        evaluator: Evaluator name (should be gemini-25-flash-lite)
        reasonableness_version: Version of reasonableness prompt (e.g., simple_reasonable_recommendation_v2 or v3)

    Returns:
        Dictionary mapping iteration number to summary data
    """
    eval_target_path = base_path / eval_target / f"evaluator-{evaluator}"

    # If reasonableness_version is specified, add it to the path
    if reasonableness_version:
        eval_target_path = eval_target_path / reasonableness_version

    if not eval_target_path.exists():
        print(f"Warning: Path does not exist: {eval_target_path}")
        return {}

    iteration_data = {}

    # Find all iteration directories
    for iteration_dir in eval_target_path.iterdir():
        if not iteration_dir.is_dir() or not iteration_dir.name.startswith("iteration-"):
            continue

        iteration_str = iteration_dir.name.replace("iteration-", "")

        # Skip 'base' iteration for now (treat as special case if needed)
        if iteration_str == "base":
            continue

        try:
            iteration_num = int(iteration_str)
        except ValueError:
            print(f"Warning: Skipping non-numeric iteration: {iteration_str}")
            continue

        # Load summary.json
        summary_path = iteration_dir / "summary.json"
        if not summary_path.exists():
            print(f"Warning: No summary.json found in {iteration_dir}")
            continue

        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            iteration_data[iteration_num] = summary
        except Exception as e:
            print(f"Error loading {summary_path}: {e}")

    return iteration_data


def compute_proportions(iteration_data: Dict[int, Dict]) -> Dict[int, Dict[str, float]]:
    """
    Compute proportions for each category across iterations.

    Returns both:
    - False negative rate: incorrect_reasonable / (incorrect_reasonable + incorrect_unreasonable)
      (Of incorrect outputs, what proportion did judge miss?)
    - False negative proportion: incorrect_reasonable / total
      (Of all datapoints, what proportion are false negatives?)

    Args:
        iteration_data: Dictionary mapping iteration to summary data

    Returns:
        Dictionary mapping iteration to proportions for each category
    """
    proportions = {}

    for iteration, summary in iteration_data.items():
        category_counts = summary['statistics']['category_counts']
        total = summary['statistics']['total_processed']

        if total == 0:
            print(f"Warning: No datapoints processed for iteration {iteration}")
            continue

        # Get counts
        incorrect_reasonable = category_counts.get('incorrect_reasonable', 0)
        incorrect_unreasonable = category_counts.get('incorrect_unreasonable', 0)

        # Total number of times model was incorrect
        total_incorrect = incorrect_reasonable + incorrect_unreasonable

        # Compute false negative rate: of all incorrect outputs, what proportion did judge miss?
        if total_incorrect > 0:
            false_negative_rate = incorrect_reasonable / total_incorrect
        else:
            false_negative_rate = 0.0

        # Compute false negative proportion: of all datapoints, what proportion are FN?
        false_negative_proportion = incorrect_reasonable / total

        proportions[iteration] = {
            'correct_reasonable': category_counts.get('correct_reasonable', 0) / total,
            'correct_unreasonable': category_counts.get('correct_unreasonable', 0) / total,
            'incorrect_reasonable_rate': false_negative_rate,  # FN rate conditioned on model being incorrect
            'incorrect_reasonable_proportion': false_negative_proportion,  # FN proportion of all datapoints
            'incorrect_unreasonable': category_counts.get('incorrect_unreasonable', 0) / total,
        }

    return proportions


def plot_single_metric(experiments_data: List[Dict], output_path: Path,
                      category_key: str, title: str, ylabel: str, eval_target_preference: str = None):
    """
    Create a single line plot for a specific metric.

    Args:
        experiments_data: List of dicts with 'experiment_name', 'proportions_dict' keys
        output_path: Path to save the plot
        category_key: Key for the metric to plot
        title: Plot title
        ylabel: Y-axis label
        eval_target_preference: Specific eval target to use (if None, uses preference order)
    """
    # Create single plot (matching plot_reward.py dimensions)
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Force serif font for this figure - more explicit approach (from plot_reward.py)
    import matplotlib as mpl
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Also try setting it directly on pyplot
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Define colors for each experiment (matching plot_reward.py order: Risky, Safe, Now, Later)
    colors = ['#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A']  # Blue, Green, Orange, Purple

    # Plot each experiment
    for idx, exp_data in enumerate(experiments_data):
        experiment_name = exp_data['experiment_name']
        proportions_dict = exp_data['proportions_dict']

        # Select which eval target to use
        if eval_target_preference and eval_target_preference in proportions_dict:
            proportions = proportions_dict[eval_target_preference]
        elif 'constitution_and_reasoning_and_response' in proportions_dict:
            proportions = proportions_dict['constitution_and_reasoning_and_response']
        elif proportions_dict:
            proportions = list(proportions_dict.values())[0]
        else:
            continue

        if not proportions:
            continue

        # Sort iterations and limit to first 9 (0-8)
        iterations = sorted([it for it in proportions.keys() if it <= 8])
        values = [proportions[it][category_key] for it in iterations]

        # Plot line (matching plot_reward.py style)
        color = colors[idx % len(colors)]
        ax.plot(iterations, values, marker='o', linewidth=2, markersize=6,
               color=color, alpha=0.7, label=experiment_name)

    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)
    ax.set_ylim(0, 1)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()

    # Save figure in multiple formats for publication
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as high-quality PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved plot to: {output_path}")

    # Also save as PDF for vector graphics
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved PDF to: {pdf_path}")

    plt.close()


def plot_reasonableness_proportions(experiments_data: List[Dict],
                                    output_path: Path,
                                    eval_target_preference: str = None):
    """
    Create line plots showing False Negative trends over iterations for multiple experiments.
    Generates two plots:
    1. False Negative Rate (conditioned on model being incorrect)
    2. False Negative Proportion (of all datapoints)

    False Negative = Model incorrect (didn't follow constitution) but judge deemed it reasonable.
    (Judge failed to catch the problem)

    Args:
        experiments_data: List of dicts with 'experiment_name', 'proportions_dict' keys
        output_path: Path to save the plots
        eval_target_preference: Specific eval target to use (if None, uses preference order)
    """
    # Plot 1: False Negative Rate (conditioned on incorrect)
    rate_path = output_path.parent / output_path.name.replace('.png', '_rate.png')
    plot_single_metric(
        experiments_data,
        rate_path,
        'incorrect_reasonable_rate',
        'CoT Monitor False Negative Rate',
        'Judge False Negative Rate',
        eval_target_preference
    )

    # Plot 2: False Negative Proportion (of all datapoints)
    proportion_path = output_path.parent / output_path.name.replace('.png', '_proportion.png')
    plot_single_metric(
        experiments_data,
        proportion_path,
        'incorrect_reasonable_proportion',
        'CoT Monitor False Negative Proportion',
        'Judge False Negative Proportion',
        eval_target_preference
    )


def main():
    parser = argparse.ArgumentParser(
        description='Plot reasonableness category proportions over iterations'
    )
    parser.add_argument('--all-experiments', action='store_true',
                       help='Plot all four experiments (now, later, safe, risky) on one plot')
    parser.add_argument('--analysis-dirs', nargs='+',
                       help='List of analysis directories to plot')
    parser.add_argument('--output-base-dir', default='plots/reasonableness_proportions',
                       help='Base output directory for plots (default: plots/reasonableness_proportions)')
    parser.add_argument('--evaluator', default='gemini-25-flash-lite',
                       help='Evaluator name (default: gemini-25-flash-lite)')
    parser.add_argument('--reasonableness-version', default=None,
                       help='Reasonableness version (e.g., simple_reasonable_recommendation_v2 or v3). If not specified, uses old directory structure.')
    parser.add_argument('--eval-target', default=None,
                       help='Specific eval target to plot (constitution_and_response or constitution_and_reasoning_and_response). If not specified, prefers constitution_and_reasoning_and_response.')

    args = parser.parse_args()

    # Define standard experiments if --all-experiments is used (matching plot_reward.py order)
    if args.all_experiments:
        analysis_dirs = [
            ('analysis_output/reasonableness/risky-09_19_182001/safe_constitutional_cot_v2', 'Risky'),
            ('analysis_output/reasonableness/safe-09_19_182118/risky_constitutional_cot_v2', 'Safe'),
            ('analysis_output/reasonableness/now-09_20_201429/later_constitutional_cot_v2', 'Now'),
            ('analysis_output/reasonableness/later-09_20_201440/now_constitutional_cot_v2', 'Later'),
        ]
    elif args.analysis_dirs:
        # Use provided directories with auto-generated names
        analysis_dirs = [(d, Path(d).parts[-2]) for d in args.analysis_dirs]
    else:
        print("Error: Must specify either --all-experiments or --analysis-dirs")
        return

    print(f"Evaluator: {args.evaluator}")
    if args.reasonableness_version:
        print(f"Reasonableness version: {args.reasonableness_version}")

    # Load data for all experiments
    experiments_data = []
    eval_targets = ['constitution_and_response', 'constitution_and_reasoning_and_response']

    for analysis_dir, experiment_name in analysis_dirs:
        analysis_path = Path(analysis_dir)

        if not analysis_path.exists():
            print(f"Warning: Analysis directory does not exist: {analysis_path}")
            continue

        print(f"\nLoading data for {experiment_name} from: {analysis_path}")

        proportions_dict = {}

        for eval_target in eval_targets:
            # Load iteration data
            iteration_data = load_iteration_data(analysis_path, eval_target, args.evaluator, args.reasonableness_version)

            if not iteration_data:
                continue

            # Compute proportions
            proportions = compute_proportions(iteration_data)
            proportions_dict[eval_target] = proportions

        if proportions_dict:
            experiments_data.append({
                'experiment_name': experiment_name,
                'proportions_dict': proportions_dict
            })
            print(f"  Loaded data for {experiment_name}")

    if not experiments_data:
        print("Error: No data found for any experiment!")
        return

    # Create output path
    output_path = Path(args.output_base_dir) / f"evaluator-{args.evaluator}"
    if args.reasonableness_version:
        output_path = output_path / args.reasonableness_version

    # Add eval_target to path if specified to avoid overwriting
    if args.eval_target:
        output_path = output_path / args.eval_target

    output_path = output_path / "combined_reasonableness_proportions.png"

    plot_reasonableness_proportions(experiments_data, output_path, args.eval_target)

    print(f"\n{'='*80}")
    print("PLOTTING COMPLETE!")
    print(f"Plot saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
