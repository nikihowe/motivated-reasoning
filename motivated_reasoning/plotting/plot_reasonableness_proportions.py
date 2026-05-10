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

        proportions[iteration] = {
            'correct_reasonable': category_counts.get('correct_reasonable', 0) / total,
            'correct_unreasonable': category_counts.get('correct_unreasonable', 0) / total,
            'incorrect_reasonable': category_counts.get('incorrect_reasonable', 0) / total,
            'incorrect_unreasonable': category_counts.get('incorrect_unreasonable', 0) / total,
        }

    return proportions


def plot_reasonableness_proportions(proportions_dict: Dict[str, Dict[int, Dict[str, float]]],
                                    output_path: Path, prompt_type: str):
    """
    Create line plots showing proportion trends over iterations.

    Args:
        proportions_dict: Dictionary mapping eval_target to iteration proportions
        output_path: Path to save the plot
        prompt_type: Prompt type name for the title
    """
    # Define categories and their display names
    categories = [
        ('correct_reasonable', 'True Positive (TP): Correct & Reasonable'),
        ('correct_unreasonable', 'False Negative (FN): Correct & Unreasonable'),
        ('incorrect_reasonable', 'False Positive (FP): Incorrect & Reasonable'),
        ('incorrect_unreasonable', 'True Negative (TN): Incorrect & Unreasonable')
    ]

    # Create figure with subplots for each category
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (category_key, category_name) in enumerate(categories):
        ax = axes[idx]

        # Plot each eval_target
        for eval_target, proportions in proportions_dict.items():
            if not proportions:
                continue

            # Sort iterations
            iterations = sorted(proportions.keys())
            values = [proportions[it][category_key] for it in iterations]

            # Create label from eval_target
            if 'reasoning' in eval_target:
                label = 'Constitution + Reasoning + Response'
            else:
                label = 'Constitution + Response'

            # Plot line
            ax.plot(iterations, values, marker='o', linewidth=2,
                   label=label, alpha=0.8)

        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title(category_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(-0.05, 1.05)

        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.suptitle(f'Reasonableness Category Proportions Over Iterations\nPrompt Type: {prompt_type}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot reasonableness category proportions over iterations'
    )
    parser.add_argument('analysis_dir', type=str,
                       help='Path to analysis directory at prompt_type level (e.g., analysis_output/reasonableness/now-09_20_201429/later_constitutional_cot_v2)')
    parser.add_argument('--output-base-dir', default='plots/reasonableness_proportions',
                       help='Base output directory for plots (default: plots/reasonableness_proportions)')
    parser.add_argument('--evaluator', default='gemini-25-flash-lite',
                       help='Evaluator name (default: gemini-25-flash-lite)')
    parser.add_argument('--reasonableness-version', default=None,
                       help='Reasonableness version (e.g., simple_reasonable_recommendation_v2 or v3). If not specified, uses old directory structure.')

    args = parser.parse_args()

    analysis_path = Path(args.analysis_dir)

    if not analysis_path.exists():
        print(f"Error: Analysis directory does not exist: {analysis_path}")
        return

    # Parse the path to extract experiment_dir and prompt_type
    # Expected path: analysis_output/reasonableness/EXPERIMENT_DIR/PROMPT_TYPE
    parts = analysis_path.parts
    try:
        reasonableness_idx = parts.index('reasonableness')
        experiment_dir = parts[reasonableness_idx + 1]
        prompt_type = parts[reasonableness_idx + 2]
    except (ValueError, IndexError):
        print(f"Warning: Could not parse experiment_dir from path. Using fallback.")
        experiment_dir = 'unknown'
        prompt_type = analysis_path.name

    print(f"Loading data from: {analysis_path}")
    print(f"Experiment: {experiment_dir}")
    print(f"Prompt type: {prompt_type}")
    print(f"Evaluator: {args.evaluator}")
    if args.reasonableness_version:
        print(f"Reasonableness version: {args.reasonableness_version}")

    # Load data for both eval targets
    eval_targets = ['constitution_and_response', 'constitution_and_reasoning_and_response']

    proportions_dict = {}

    for eval_target in eval_targets:
        print(f"\nProcessing {eval_target}...")

        # Load iteration data
        iteration_data = load_iteration_data(analysis_path, eval_target, args.evaluator, args.reasonableness_version)

        if not iteration_data:
            print(f"  No data found for {eval_target}")
            continue

        # Verify evaluator
        for iteration, summary in iteration_data.items():
            actual_evaluator = summary['experiment_config']['evaluator']
            if actual_evaluator != args.evaluator:
                raise ValueError(
                    f"Expected evaluator '{args.evaluator}' but found '{actual_evaluator}' "
                    f"in iteration {iteration} of {eval_target}"
                )

        print(f"  Loaded {len(iteration_data)} iterations")

        # Compute proportions
        proportions = compute_proportions(iteration_data)
        proportions_dict[eval_target] = proportions

        print(f"  Computed proportions for {len(proportions)} iterations")

    if not proportions_dict:
        print("Error: No data found to plot!")
        return

    # Determine which eval_target has data (prefer constitution_and_reasoning_and_response)
    if 'constitution_and_reasoning_and_response' in proportions_dict:
        eval_target_for_path = 'constitution_and_reasoning_and_response'
    else:
        eval_target_for_path = list(proportions_dict.keys())[0]

    # Create output path with directory structure
    # plots/EXPERIMENT_DIR/PROMPT_TYPE/EVAL_TARGET/evaluator-EVALUATOR/REASONABLENESS_VERSION/reasonableness_proportions.png
    output_path = Path(args.output_base_dir) / experiment_dir / prompt_type / eval_target_for_path / f"evaluator-{args.evaluator}"

    if args.reasonableness_version:
        output_path = output_path / args.reasonableness_version

    output_path = output_path / "reasonableness_proportions.png"

    plot_reasonableness_proportions(proportions_dict, output_path, prompt_type)

    print(f"\n{'='*80}")
    print("PLOTTING COMPLETE!")
    print(f"Plot saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
