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

eval_target_to_label = {
    'response': 'Refusal Rate',
    'reasoning': 'Mean Motivated Reasoning Score',
    'full': 'Full Response Accuracy',
    'everything': 'Full Response Accuracy',
    'system_and_reasoning': 'System + Reasoning Score',
    'system_and_response': 'System + Response Score',
    'constitution_and_reasoning': 'Constitution + Reasoning Score',
    'constitution_and_response': 'Constitution + Response Score',
    'constitution_and_reasoning_and_response': 'Constitution + Full Score'
}


def find_experiment_directories(evaluation_output_dir: str) -> List[str]:
    """
    Find all experiment directories in the evaluation_output directory.

    Args:
        evaluation_output_dir: Path to the evaluation_output directory

    Returns:
        List of experiment directory paths
    """
    experiment_dirs = []
    eval_path = Path(evaluation_output_dir)

    if not eval_path.exists():
        print(f"Directory {evaluation_output_dir} does not exist")
        return experiment_dirs

    # Find all subdirectories that contain evaluation data
    for item in eval_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Skip the 'old' directory as it seems to contain archived experiments
            if item.name != 'old':
                experiment_dirs.append(str(item))

    return sorted(experiment_dirs)


def extract_evaluation_scores(experiment_dir: str, evaluator_only: bool = True) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Extract evaluation scores across iterations for all experiments in a directory.
    Only processes experiments that use simple_compliance evaluation.

    Args:
        experiment_dir: Path to an experiment directory
        evaluator_only: If True, only include experiments with evaluator-based structure

    Returns:
        Dictionary with structure: {experiment_name: {eval_target: [(iteration, score), ...]}}
    """
    results = {}
    exp_path = Path(experiment_dir)

    # Walk through the new directory structure: prompt_type/eval_target/evaluator/suffix/iteration/
    for prompt_type_dir in exp_path.iterdir():
        if not prompt_type_dir.is_dir():
            continue

        prompt_type = prompt_type_dir.name

        for eval_target_dir in prompt_type_dir.iterdir():
            if not eval_target_dir.is_dir():
                continue

            eval_target = eval_target_dir.name

            for evaluator_dir in eval_target_dir.iterdir():
                if not evaluator_dir.is_dir() or not evaluator_dir.name.startswith("evaluator-"):
                    continue

                if evaluator_only and not evaluator_dir.name.startswith("evaluator-"):
                    continue

                evaluator_name = evaluator_dir.name.replace("evaluator-", "")

                # Get all suffix directories and sort by modification time to use most recent
                suffix_dirs = [d for d in evaluator_dir.iterdir() if d.is_dir()]
                if not suffix_dirs:
                    continue

                # Only process simple_compliance evaluations
                compliance_dirs = [d for d in suffix_dirs if d.name == 'simple_compliance']
                if not compliance_dirs:
                    continue

                # Use the most recently modified simple_compliance directory
                latest_suffix_dir = max(compliance_dirs, key=lambda d: d.stat().st_mtime)
                suffix_name = latest_suffix_dir.name

                # Create experiment name from path components
                experiment_name = f"{prompt_type}/{eval_target}/{evaluator_name}/{suffix_name}"

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

                            # Extract scores from the data using the new 'evaluator_score' field
                            if isinstance(data, list) and len(data) > 0:
                                scores = []
                                for item in data:
                                    # In new format, score is in 'evaluator_score' field
                                    if 'evaluator_score' in item and item['evaluator_score'] is not None:
                                        scores.append(item['evaluator_score'])

                                if scores:  # Only add if we have valid scores
                                    # Calculate refusal rate as 1 - compliance
                                    compliance_rate = np.mean(scores)
                                    refusal_rate = 1 - compliance_rate

                                    # Initialize nested dictionaries if needed
                                    if experiment_name not in results:
                                        results[experiment_name] = {}
                                    if eval_target not in results[experiment_name]:
                                        results[experiment_name][eval_target] = []

                                    results[experiment_name][eval_target].append((iteration_num, refusal_rate))

                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Error reading {latest_file}: {e}")
                            continue

    # Sort iterations for each experiment and eval_target
    for exp_name in results:
        for eval_target in results[exp_name]:
            results[exp_name][eval_target].sort(key=lambda x: x[0])

    return results


def plot_evaluation_scores(results: Dict[str, Dict[str, List[Tuple[int, float]]]],
                          experiment_dir_name: str,
                          output_dir: str = "plots",
                          experiment_path: str = None) -> None:
    """
    Plot refusal rates over iterations for all experiments.
    Creates separate plots for each subdirectory to preserve the structure.

    Args:
        results: Dictionary with evaluation scores
        experiment_dir_name: Name of the experiment directory (for titles)
        output_dir: Directory to save plots
        experiment_path: Full path to the experiment directory (for preserving structure)
    """
    # Create base output directory preserving the full structure
    if experiment_path:
        # Extract the relative path from evaluation_output to preserve structure
        eval_output_dir = "/nas/ucb/nikihowe/motivated-reasoning/evaluation_output"
        if experiment_path.startswith(eval_output_dir):
            rel_path = os.path.relpath(experiment_path, eval_output_dir)
            base_output_dir = os.path.join(output_dir, rel_path)
        else:
            base_output_dir = os.path.join(output_dir, experiment_dir_name)
    else:
        base_output_dir = os.path.join(output_dir, experiment_dir_name)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Group results by model type (more specific grouping)
    subdir_groups = {}
    for exp_name, exp_data in results.items():
        # Split the experiment name to get a more meaningful grouping
        path_parts = exp_name.split('/')
        if len(path_parts) >= 2:
            # Use first two parts for more specific grouping
            # e.g., 'risky_constitutional_cot/response' vs 'risky_constitutional_cot/reasoning'
            subdir = '/'.join(path_parts[:2])  # e.g., 'risky_constitutional_cot/response'
        elif len(path_parts) >= 1:
            subdir = path_parts[0]  # fallback to first part only
        else:
            continue

        if subdir not in subdir_groups:
            subdir_groups[subdir] = {}
        subdir_groups[subdir][exp_name] = exp_data

    # Create plots for each subdirectory
    for subdir, subdir_results in subdir_groups.items():
        # Create subdirectory output path
        subdir_output_dir = os.path.join(base_output_dir, subdir)
        os.makedirs(subdir_output_dir, exist_ok=True)

        # Get all unique eval targets for this subdirectory
        all_eval_targets = set()
        for exp_data in subdir_results.values():
            all_eval_targets.update(exp_data.keys())

        # Create plots for each eval target in this subdirectory
        for eval_target in all_eval_targets:
            fig, ax = plt.subplots(figsize=(3.5, 3))

            # Force serif font for this figure - more explicit approach
            import matplotlib as mpl
            mpl.rcParams['font.family'] = ['serif']
            mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

            # Also try setting it directly on pyplot
            plt.rcParams['font.family'] = ['serif']
            plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

            # Plot each experiment in this subdirectory
            for exp_name, exp_data in subdir_results.items():
                if eval_target in exp_data and exp_data[eval_target]:
                    # Limit to first 11 iterations (base + 0-10)
                    limited_data = exp_data[eval_target][:11]
                    iterations, scores = zip(*limited_data)

                    # Convert base iteration (-1) to a more readable label, shift numeric iterations by 1
                    iterations_display = [f"base" if i == -1 else str(i + 1) for i in iterations]

                    # Use the same red color as HarmBench in plot_reward.py
                    color = '#E31A1C'  # Red
                    ax.plot(range(len(iterations)), scores, marker='o', linewidth=2, markersize=6, label=exp_name, alpha=0.8, color=color)

            ax.set_xlabel('RL Training Iteration')
            ax.set_ylabel(f'{eval_target_to_label.get(eval_target, eval_target.title())}')

            human_preference = nice_format(base_output_dir.split('/')[-1])
            constitution = nice_format(subdir.split('/')[0])
            # ax.set_title(f'Preferences: {human_preference}\nConstitution: {constitution}')
            ax.set_title(f'HarmBench')
            ax.grid(True, alpha=0.3)

            # Set y-axis range for specific eval targets
            if eval_target == 'reasoning':
                ax.set_ylim(1, 5)
            elif eval_target == 'response':
                ax.set_ylim(0, 1)  # Refusal rate is 0-1

            # Set x-axis labels
            if subdir_results:  # If we have any results
                sample_exp = next(iter(subdir_results.values()))
                if eval_target in sample_exp:
                    sample_iterations, _ = zip(*sample_exp[eval_target])
                    iterations_display = [f"base" if i == -1 else str(i + 1) for i in sample_iterations]
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

            # Make sure legend uses serif font too
            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_fontfamily('serif')

            plt.tight_layout()

            # Save the plot
            filename = f"refusal_over_iterations.png"
            filepath = os.path.join(subdir_output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")

            plt.show()
            plt.close()


def plot_refusal(evaluation_output_dir: str, output_dir: str = "plots") -> None:
    """
    Main function to plot refusal rates for all experiments in the evaluation_output directory.

    Args:
        evaluation_output_dir: Path to the evaluation_output directory
        output_dir: Directory to save plots
    """
    print(f"Looking for experiments in: {evaluation_output_dir}")

    # Find all experiment directories
    experiment_dirs = find_experiment_directories(evaluation_output_dir)
    print(f"Found {len(experiment_dirs)} experiment directories")

    # Process each experiment directory
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"\\nProcessing experiment: {exp_name}")

        # Extract evaluation scores (evaluator-based only, simple_compliance only)
        results = extract_evaluation_scores(exp_dir, evaluator_only=True)

        if results:
            print(f"Found {len(results)} sub-experiments with compliance evaluation data")
            # Create plots with full path for structure preservation
            plot_evaluation_scores(results, exp_name, output_dir, exp_dir)
        else:
            print(f"No compliance evaluation data found in {exp_dir}")


def plot_specific_experiment(experiment_dir: str, output_dir: str = "plots") -> None:
    """
    Plot refusal rates for a specific experiment directory.

    Args:
        experiment_dir: Path to a specific experiment directory (e.g., evaluation_output/safe-09_19_182118)
        output_dir: Directory to save plots
    """
    import os

    if not os.path.exists(experiment_dir):
        print(f"Directory {experiment_dir} does not exist")
        return

    exp_name = os.path.basename(experiment_dir)
    print(f"Processing specific experiment: {exp_name}")

    # Extract evaluation scores (evaluator-based only, simple_compliance only)
    results = extract_evaluation_scores(experiment_dir, evaluator_only=True)

    if results:
        print(f"Found {len(results)} sub-experiments with compliance evaluation data")
        # Create plots with directory structure preservation
        plot_evaluation_scores(results, exp_name, output_dir, experiment_dir)
    else:
        print(f"No compliance evaluation data found in {experiment_dir}")


def plot_experiment_by_name(experiment_name: str, evaluation_output_dir: str = "/nas/ucb/nikihowe/motivated-reasoning/evaluation_output", output_dir: str = "plots") -> None:
    """
    Plot refusal rates for a specific experiment by name.

    Args:
        experiment_name: Name of the experiment (e.g., 'risky-09_19_182001', 'safe-09_19_182118')
        evaluation_output_dir: Path to the evaluation_output directory
        output_dir: Directory to save plots
    """
    import os

    # Construct the full path to the experiment
    experiment_path = os.path.join(evaluation_output_dir, experiment_name)

    if not os.path.exists(experiment_path):
        print(f"Experiment '{experiment_name}' not found in {evaluation_output_dir}")
        print(f"Available experiments:")
        try:
            available_experiments = find_experiment_directories(evaluation_output_dir)
            for exp_dir in available_experiments:
                exp_name = os.path.basename(exp_dir)
                print(f"  - {exp_name}")
        except Exception as e:
            print(f"  Error listing experiments: {e}")
        return

    print(f"Plotting experiment: {experiment_name}")
    plot_specific_experiment(experiment_path, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot refusal rates over iterations')
    parser.add_argument('experiment', nargs='?', type=str,
                       help='Specific experiment name to plot (e.g., risky-09_19_182001)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Plot all experiments (default behavior)')
    parser.add_argument('--evaluation-output-dir', '-d',
                       default="/nas/ucb/nikihowe/motivated-reasoning/evaluation_output",
                       help='Path to evaluation_output directory')
    parser.add_argument('--output-dir', '-o', default='plots',
                       help='Output directory for plots')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available experiments')

    args = parser.parse_args()

    if args.list:
        # List available experiments
        experiment_dirs = find_experiment_directories(args.evaluation_output_dir)
        print("Available experiments:")
        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            print(f"  - {exp_name}")
    elif args.experiment:
        # Plot specific experiment
        plot_experiment_by_name(args.experiment, args.evaluation_output_dir, args.output_dir)
    else:
        # Plot all experiments (default)
        plot_refusal(args.evaluation_output_dir, args.output_dir)