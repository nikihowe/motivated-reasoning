import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from motivated_reasoning.plotting.plotting_utils import nice_format

# Set style for publication-ready plots
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

# Define the four specific experiments we want to combine
# Ordered by preference (before arrow): Risky, Safe, Now, Later -> Blue, Green, Orange, Purple
EXPERIMENT_CONFIGS = {
    'risky-09_19_182001': {
        'inference_prompt_dir': 'safe_constitutional_cot_v2',
        'label': 'Risky → Safe'
    },
    'safe-09_19_182118': {
        'inference_prompt_dir': 'risky_constitutional_cot_v2',
        'label': 'Safe → Risky'
    },
    'now-09_20_201429': {
        'inference_prompt_dir': 'later_constitutional_cot_v2',
        'label': 'Now → Later'
    },
    'later-09_20_201440': {
        'inference_prompt_dir': 'now_constitutional_cot_v2',
        'label': 'Later → Now'
    }
}


def extract_evaluation_scores_for_experiment(experiment_dir: str, inference_prompt_dir: str, eval_target: str = 'response') -> List[Tuple[int, float]]:
    """
    Extract evaluation scores for a specific experiment configuration.

    Args:
        experiment_dir: Path to the experiment directory
        inference_prompt_dir: Name of the inference prompt directory
        eval_target: Evaluation target (default: 'response')

    Returns:
        List of (iteration, score) tuples
    """
    results = []
    exp_path = Path(experiment_dir)

    if not exp_path.exists():
        print(f"Warning: Experiment directory {experiment_dir} does not exist")
        return results

    # Navigate to the specific path: experiment/inference_prompt_dir/eval_target/
    target_path = exp_path / inference_prompt_dir / eval_target

    if not target_path.exists():
        print(f"Warning: Path {target_path} does not exist")
        return results

    # Look for evaluator directories
    for evaluator_dir in target_path.iterdir():
        if not evaluator_dir.is_dir() or not evaluator_dir.name.startswith("evaluator-"):
            continue

        # Get all suffix directories and use the most recent
        suffix_dirs = [d for d in evaluator_dir.iterdir() if d.is_dir()]
        if not suffix_dirs:
            continue

        latest_suffix_dir = max(suffix_dirs, key=lambda d: d.stat().st_mtime)

        # Process iterations in this suffix directory
        for iteration_dir in latest_suffix_dir.iterdir():
            if not iteration_dir.is_dir() or not iteration_dir.name.startswith("iteration-"):
                continue

            # Extract iteration number
            iteration_str = iteration_dir.name.replace("iteration-", "")
            if iteration_str == "base":
                iteration_num = -1  # Use -1 for base iteration
            else:
                try:
                    iteration_num = int(iteration_str)
                except ValueError:
                    continue

            # Find the most recent evaluation JSON file
            eval_files = list(iteration_dir.glob("*eval*.json"))
            if not eval_files:
                continue

            latest_file = max(eval_files, key=lambda f: f.stat().st_mtime)

            # Load and parse the JSON file
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)

                # Extract scores using the new 'evaluator_score' field
                if isinstance(data, list) and len(data) > 0:
                    scores = []
                    for item in data:
                        if 'evaluator_score' in item and item['evaluator_score'] is not None:
                            scores.append(item['evaluator_score'])

                    if scores:  # Only add if we have valid scores
                        avg_score = np.mean(scores)
                        results.append((iteration_num, avg_score))

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {latest_file}: {e}")
                continue

    # Sort by iteration number
    results.sort(key=lambda x: x[0])
    return results


def create_four_way_plot(evaluation_output_dir: str, output_dir: str = "plots") -> None:
    """
    Create a combined plot showing all four constitutional experiments.

    Args:
        evaluation_output_dir: Path to the evaluation_output directory
        output_dir: Directory to save plots
    """
    print(f"Creating four-way constitutional comparison plot...")

    # Create output directory
    output_path = Path(output_dir) / "combined"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Force serif font for this figure - more explicit approach
    import matplotlib as mpl
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Also try setting it directly on pyplot
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Set style to match plot_simple_evaluation.py
    plt.style.use('default')
    # Colors chosen for maximum distinguishability (matching plot_reward.py, excluding HarmBench)
    colors = ['#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A']  # Blue, Green, Orange, Purple

    experiment_data = {}

    # Extract data for each experiment
    for exp_name, config in EXPERIMENT_CONFIGS.items():
        exp_dir = Path(evaluation_output_dir) / exp_name

        print(f"Processing {exp_name}: {config['label']}")

        scores = extract_evaluation_scores_for_experiment(
            str(exp_dir),
            config['inference_prompt_dir'],
            eval_target='response'
        )

        if scores:
            # Limit to first 11 iterations (base + 0-10)
            limited_scores = scores[:11]
            experiment_data[config['label']] = limited_scores
            print(f"  Found {len(limited_scores)} iterations")
        else:
            print(f"  No data found for {exp_name}")

    if not experiment_data:
        print("No data found for any experiments")
        return

    # Plot each experiment
    for i, (label, data) in enumerate(experiment_data.items()):
        iterations, scores = zip(*data)

        # Convert base iteration (-1) to more readable label, shift numeric iterations by 1
        iterations_display = [0 if i == -1 else i + 1 for i in iterations]

        color = colors[i % len(colors)]
        ax.plot(range(len(iterations)), scores, marker='o', linewidth=2, markersize=6,
               color=color, label=label, alpha=0.8)

    # Customize the plot
    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel('Constitution Following Rate')
    ax.set_title('Constitution Following', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis range for response evaluation (0 to 1)
    ax.set_ylim(0, 1)

    # Set x-axis labels
    if experiment_data:
        # Get a sample experiment to determine x-axis structure
        sample_data = next(iter(experiment_data.values()))
        sample_iterations, _ = zip(*sample_data)
        iterations_display = ["base" if i == -1 else str(i + 1) for i in sample_iterations]
        # Limit to first 11 iterations (base + 0-10)
        iterations_display = iterations_display[:11]
        ax.set_xticks(range(len(iterations_display)))
        ax.set_xticklabels(iterations_display, fontsize=7)

        # Remove x-axis padding to match plot_simple_evaluation.py
        ax.set_xlim(0, len(iterations_display) - 1)

        # Explicitly set both x and y axis tick label font sizes
        ax.tick_params(axis='both', labelsize=7)

    # Set serif font on all text elements
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('serif')
    ax.xaxis.label.set_fontfamily('serif')
    ax.yaxis.label.set_fontfamily('serif')
    ax.title.set_fontfamily('serif')

    # Add legend with font size one smaller than axis labels
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=8)
    for text in legend.get_texts():
        text.set_fontfamily('serif')

    plt.tight_layout()

    # Save the plot
    filename = "response_constitutional_cot_v2_four_way.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")

    plt.show()
    plt.close()


def main():
    """Main function to create the four-way constitutional comparison plot."""
    import argparse

    parser = argparse.ArgumentParser(description='Create four-way constitutional comparison plot')
    parser.add_argument('--evaluation-output-dir', '-d',
                       default="/nas/ucb/nikihowe/motivated-reasoning/evaluation_output",
                       help='Path to evaluation_output directory')
    parser.add_argument('--output-dir', '-o', default='plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    create_four_way_plot(args.evaluation_output_dir, args.output_dir)


if __name__ == "__main__":
    main()