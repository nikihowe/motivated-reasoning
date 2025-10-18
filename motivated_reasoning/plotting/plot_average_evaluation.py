import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
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


def load_evaluation_results_by_suffix(evaluation_dir, evaluator_name="base", prompt_type="cot_prompt", eval_target="reasoning"):
    """
    Load evaluation results organized by suffix condition.
    Args:
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator to load
        prompt_type (str): Prompt type to load (e.g., "constitutional_cot", "simple_cot")
        eval_target (str): Evaluation target to load (e.g., "full", "reasoning", "response")
    Returns:
        dict: Dictionary mapping suffix names to results by iteration
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir / prompt_type / eval_target / f"evaluator-{evaluator_name}"

    if not evaluation_path.exists():
        print(f"Error: Evaluation path {evaluation_path} does not exist")
        return {}

    results_by_suffix = {}

    # Look for suffix directories
    suffix_dirs = [d for d in evaluation_path.iterdir() if d.is_dir()]
    if not suffix_dirs:
        print(f"Error: No suffix directories found in {evaluation_path}")
        return {}

    print(f"Found {len(suffix_dirs)} suffix directories for evaluator-{evaluator_name}")

    for suffix_dir in suffix_dirs:
        if not suffix_dir.is_dir():
            continue

        suffix_name = suffix_dir.name
        results_by_iteration = {}

        # Find all iteration directories under the suffix directory
        iteration_dirs = [d for d in suffix_dir.iterdir()
                         if d.is_dir() and d.name.startswith("iteration-")]

        if not iteration_dirs:
            print(f"Warning: No iteration directories found in suffix {suffix_name}")
            continue

        print(f"Found {len(iteration_dirs)} iteration directories for suffix {suffix_name}")

        # Look for iteration directories within the suffix
        for iteration_dir in iteration_dirs:
            # Extract iteration identifier from directory name
            iteration_str = iteration_dir.name.split("-")[1]
            # Keep as string to handle both "base" and numeric iterations
            iteration = iteration_str

            # Look for eval JSON files in this iteration
            json_files = list(iteration_dir.glob("*eval*.json"))
            if not json_files:
                continue

            # Load the most recent JSON file (by modification time)
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

            try:
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                results_by_iteration[iteration] = results
                print(f"  Loaded {len(results)} evaluation examples for {suffix_name} ({prompt_type}) iteration {iteration}")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
                continue

        if results_by_iteration:
            results_by_suffix[suffix_name] = results_by_iteration

    return results_by_suffix


def calculate_average_scores(results_by_suffix):
    """
    Calculate mean and median motivated reasoning scores for each iteration.
    Args:
        results_by_suffix: Dictionary mapping suffix names to results by iteration
    Returns:
        dict: Dictionary with structure {suffix: {iteration: {'mean': float, 'median': float, 'scores': list}}}
    """
    aggregated_results = {}

    for suffix_name, results_by_iteration in results_by_suffix.items():
        suffix_stats = {}

        for iteration, results in results_by_iteration.items():
            # Extract all evaluator_score values from this iteration
            scores = []
            for item in results:
                if 'evaluator_score' in item and item['evaluator_score'] is not None:
                    score = item['evaluator_score']
                    if score != -1:  # Exclude "no answer" scores
                        scores.append(score)

            if scores:  # Only calculate if we have valid scores
                suffix_stats[iteration] = {
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'scores': scores,
                    'count': len(scores)
                }
                print(f"    Iteration {iteration}: {len(scores)} scores, mean={np.mean(scores):.2f}, median={np.median(scores):.2f}")

        if suffix_stats:
            aggregated_results[suffix_name] = suffix_stats

    return aggregated_results


def plot_combined_scores(aggregated_results, stat_type='mean'):
    """
    Plot average motivated reasoning scores over iterations.
    Args:
        aggregated_results: Dictionary with calculated statistics
        evaluation_dir: Name of the evaluation directory
        prompt_type: Prompt type used
        eval_target: Evaluation target
        evaluator_name: Evaluator name
        stat_type: Either 'mean' or 'median'
    """
    # Create figure with same style as plot_reward.py
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Force serif font
    import matplotlib as mpl
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Colors for different suffix conditions (same as plot_reward.py)
    colors = ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A']  # Red, Blue, Green, Orange, Purple

    # Sort iterations function (same as plot_evaluation.py)
    def sort_iterations(iteration_key):
        if iteration_key == "base":
            return -1  # "base" comes first
        else:
            try:
                return int(iteration_key)
            except ValueError:
                return float('inf')  # Unknown iterations go last

    # Plot each run (now using custom labels)
    for i, (label, suffix_stats) in enumerate(aggregated_results.items()):
        # Sort iterations
        sorted_iterations = sorted(suffix_stats.keys(), key=sort_iterations)

        # Limit to first 11 iterations (base + 0-10)
        sorted_iterations = sorted_iterations[:11]

        if not sorted_iterations:
            continue

        # Extract data for plotting
        iterations_display = []
        scores = []

        for iteration in sorted_iterations:
            if iteration == "base":
                iterations_display.append("base")
            else:
                try:
                    numeric_iter = int(iteration)
                    iterations_display.append(str(numeric_iter + 1))  # Shift by 1 for display
                except ValueError:
                    iterations_display.append(iteration)

            scores.append(suffix_stats[iteration][stat_type])

        # Plot the data
        color = colors[i % len(colors)]
        ax.plot(range(len(iterations_display)), scores, marker='o', linewidth=2, markersize=6,
                color=color, alpha=0.7, label=label)

    # Customize plot
    ax.set_xlabel('RL Training Iteration')
    # Set y-label with smaller second line
    ylabel_main = f'{stat_type.title()} Motivated Reasoning Score'
    ylabel_small = '(1=Fully Genuine, 5=Fully Motivated)'
    ax.set_ylabel(f'{ylabel_main}\n{ylabel_small}')

    # Make the second line smaller by modifying the text after setting
    ylabel_text = ax.get_ylabel()
    ax.set_ylabel('')  # Clear it first
    ax.text(-0.15, 0.5, ylabel_main, transform=ax.transAxes, rotation=90,
            va='center', ha='center', fontsize=9, fontfamily='serif')
    ax.text(-0.1, 0.5, ylabel_small, transform=ax.transAxes, rotation=90,
            va='center', ha='center', fontsize=7, fontfamily='serif')

    # Create title
    ax.set_title('All Tasks Motivated Reasoning')

    # Set y-axis range (motivated reasoning scores are 1-5)
    ax.set_ylim(1, 5)

    # Set x-axis
    if aggregated_results:  # If we have any results
        # Use the first suffix to get iteration labels
        first_suffix = next(iter(aggregated_results.values()))
        sorted_iterations = sorted(first_suffix.keys(), key=sort_iterations)[:11]
        iterations_display = []
        for iteration in sorted_iterations:
            if iteration == "base":
                iterations_display.append("base")
            else:
                try:
                    numeric_iter = int(iteration)
                    iterations_display.append(str(numeric_iter + 1))
                except ValueError:
                    iterations_display.append(iteration)

        ax.set_xticks(range(len(iterations_display)))
        ax.set_xticklabels(iterations_display, fontsize=7, rotation=0)
        ax.set_xlim(0, len(iterations_display) - 1)

    # Style adjustments
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=7)

    # Set serif font on all text elements
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('serif')
    ax.xaxis.label.set_fontfamily('serif')
    ax.yaxis.label.set_fontfamily('serif')
    ax.title.set_fontfamily('serif')

    # Add legend if multiple conditions
    if len(aggregated_results) > 1:
        legend = ax.legend(fontsize=7, loc='best', frameon=True, fancybox=True, shadow=False)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_fontfamily('serif')

    plt.tight_layout()

    # Save the plot
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_path = plots_dir / f"combined_average_{stat_type}_motivated_reasoning.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    # Also save as PDF
    pdf_path = plots_dir / f"combined_average_{stat_type}_motivated_reasoning.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f"Saved {stat_type} plot: {output_path}")
    plt.close()


def plot_average_evaluation():
    """
    Main function to create average motivated reasoning score plots for hardcoded runs.
    """
    # Hardcoded configuration for the specific runs you want
    runs_config = [
        {
            'evaluation_dir': 'harmbench_cot_tags-08_18_162853',
            'prompt_type': 'constitutional_cot',
            'eval_target': 'constitution_and_reasoning',
            'evaluator_name': 'gemini-25-flash-lite',
            'suffix_name': 'copy_constitution_motivated_reasoning_v3',
            'label': 'HarmBench'
        },
        {
            'evaluation_dir': 'risky-09_19_182001',
            'prompt_type': 'safe_constitutional_cot_v2',
            'eval_target': 'constitution_and_reasoning',
            'evaluator_name': 'gemini-25-flash-lite',
            'suffix_name': 'copy_constitution_motivated_reasoning_v3',
            'label': 'Risky'
        },
        {
            'evaluation_dir': 'safe-09_19_182118',
            'prompt_type': 'risky_constitutional_cot_v2',
            'eval_target': 'constitution_and_reasoning',
            'evaluator_name': 'gemini-25-flash-lite',
            'suffix_name': 'copy_constitution_motivated_reasoning_v3',
            'label': 'Safe'
        },
        {
            'evaluation_dir': 'now-09_20_201429',
            'prompt_type': 'later_constitutional_cot_v2',
            'eval_target': 'constitution_and_reasoning',
            'evaluator_name': 'gemini-25-flash-lite',
            'suffix_name': 'copy_constitution_motivated_reasoning_v3',
            'label': 'Now'
        },
        {
            'evaluation_dir': 'later-09_20_201440',
            'prompt_type': 'now_constitutional_cot_v2',
            'eval_target': 'constitution_and_reasoning',
            'evaluator_name': 'gemini-25-flash-lite',
            'suffix_name': 'copy_constitution_motivated_reasoning_v3',
            'label': 'Later'
        }
    ]

    print("Loading evaluation results for hardcoded runs...")

    # Load and aggregate results for all runs
    all_aggregated_results = {}

    for run in runs_config:
        print(f"\nProcessing {run['label']}...")

        # Load results for this specific run
        results_by_suffix = load_evaluation_results_by_suffix(
            run['evaluation_dir'],
            run['evaluator_name'],
            run['prompt_type'],
            run['eval_target']
        )

        # Extract only the specific suffix we want
        if run['suffix_name'] in results_by_suffix:
            single_suffix_results = {run['suffix_name']: results_by_suffix[run['suffix_name']]}

            # Calculate average scores for this run
            run_aggregated = calculate_average_scores(single_suffix_results)

            # Store with the custom label
            if run['suffix_name'] in run_aggregated:
                all_aggregated_results[run['label']] = run_aggregated[run['suffix_name']]
                print(f"  Successfully loaded {run['label']} data")
            else:
                print(f"  Warning: No valid scores found for {run['label']}")
        else:
            print(f"  Warning: Suffix '{run['suffix_name']}' not found for {run['label']}")
            if results_by_suffix:
                print(f"    Available suffixes: {list(results_by_suffix.keys())}")

    if not all_aggregated_results:
        print("No valid data found for any runs!")
        return

    print(f"\nSuccessfully loaded data for: {list(all_aggregated_results.keys())}")

    # Create both mean and median plots
    print("Creating mean score plot...")
    plot_combined_scores(all_aggregated_results, 'mean')

    print("Creating median score plot...")
    plot_combined_scores(all_aggregated_results, 'median')

    print("Average evaluation plots completed!")


if __name__ == "__main__":
    plot_average_evaluation()