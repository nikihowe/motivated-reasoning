import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from motivated_reasoning.plotting.plotting_utils import nice_format, get_correctness_for_datapoint

# Set style for publication-ready plots
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 6,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.linewidth': 0.8,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.8,
    'lines.markersize': 4.5,
    'xtick.major.size': 3,
    'ytick.minor.size': 2,
    'ytick.major.size': 3,
    'ytick.minor.size': 2
})

plt.style.use('seaborn-v0_8-whitegrid')

# Labels for reasonableness evaluation scores
score_type_to_label = {
    'evaluator_score': 'Reasonableness Score',
    'evaluator_result': 'Evaluation Result'
}


def find_experiment_directories(evaluation_output_dir: str) -> List[str]:
    """
    Find all experiment directories that contain simple_reasonable_recommendation evaluations.
    
    Args:
        evaluation_output_dir: Path to the evaluation_output directory
        
    Returns:
        List of experiment directory paths that have reasonableness evaluations
    """
    experiment_dirs = []
    eval_path = Path(evaluation_output_dir)
    
    if not eval_path.exists():
        print(f"Warning: Evaluation output directory {evaluation_output_dir} does not exist")
        return experiment_dirs
    
    # Look for experiment directories
    for exp_dir in eval_path.iterdir():
        if exp_dir.is_dir():
            # Check if this experiment has simple_reasonable_recommendation evaluations
            has_reasonableness = False
            for root, dirs, files in os.walk(exp_dir):
                if 'simple_reasonable_recommendation' in root:
                    has_reasonableness = True
                    break
            
            if has_reasonableness:
                experiment_dirs.append(str(exp_dir))
    
    return sorted(experiment_dirs)


def extract_reasonableness_scores(experiment_dir: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Extract reasonableness evaluation scores across iterations for an experiment.
    
    Args:
        experiment_dir: Path to an experiment directory
        
    Returns:
        Dictionary with structure: {config_key: {score_type: [(iteration, score), ...], 'path_info': {...}}}
    """
    results = {}
    exp_path = Path(experiment_dir)
    exp_name = exp_path.name
    
    # Walk through the directory structure to find simple_reasonable_recommendation evaluation JSON files
    for root, dirs, files in os.walk(experiment_dir):
        # Only process directories that contain simple_reasonable_recommendation
        if 'simple_reasonable_recommendation' not in root:
            continue
            
        # Filter for evaluation JSON files
        eval_files = [f for f in files if f.startswith('eval_') and f.endswith('.json')]
        
        if eval_files:
            # Sort by filename (which contains timestamps) and take the most recent one
            eval_files.sort(reverse=True)  # Sort in descending order to get most recent first
            file = eval_files[0]  # Take the most recent file
            file_path = os.path.join(root, file)
            
            # Extract path components to identify the experiment configuration
            rel_path = os.path.relpath(root, experiment_dir)
            path_parts = rel_path.split(os.sep)
            
            # Expected structure: inference_prompt_dir/eval_target/evaluator/eval_prompt_dir/iteration-X
            if len(path_parts) >= 5:
                inference_prompt = path_parts[0]
                eval_target = path_parts[1]
                evaluator = path_parts[2]
                eval_prompt = path_parts[3]
                iteration_dir = path_parts[4]
                
                # Extract iteration number from "iteration-X" or "iteration-base"
                if iteration_dir.startswith('iteration-'):
                    iteration_str = iteration_dir.replace('iteration-', '')
                    if iteration_str == 'base':
                        iteration = -1  # Use -1 for base model
                    else:
                        try:
                            iteration = int(iteration_str)
                        except ValueError:
                            continue
                else:
                    continue
                
                # Create a unique key for this configuration that preserves the original names
                config_key = f"{inference_prompt}|{eval_target}|{evaluator}"
                
                try:
                    # Load the evaluation results
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract scores, excluding -1 (no answer) scores
                    valid_scores = []
                    total_examples = 0
                    for item in data:
                        if 'evaluator_score' in item and item['evaluator_score'] is not None:
                            total_examples += 1
                            score = item['evaluator_score']
                            if score != -1:  # Exclude "no answer" scores
                                valid_scores.append(score)
                    
                    if valid_scores:
                        # Calculate mean score for this iteration (only from valid responses)
                        mean_score = np.mean(valid_scores)
                        n_valid = len(valid_scores)
                        n_total = total_examples
                        
                        # Store the result with path information and response counts
                        if config_key not in results:
                            results[config_key] = {
                                'path_info': {
                                    'inference_prompt': inference_prompt,
                                    'eval_target': eval_target,
                                    'evaluator': evaluator,
                                    'eval_prompt': eval_prompt
                                }
                            }
                        if 'evaluator_score' not in results[config_key]:
                            results[config_key]['evaluator_score'] = []
                        
                        # Store (iteration, mean_score, n_valid, n_total)
                        results[config_key]['evaluator_score'].append((iteration, mean_score, n_valid, n_total))
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Error processing {file_path}: {e}")
                    continue
    
    # Sort iterations for each configuration
    for config_key in results:
        if 'evaluator_score' in results[config_key]:
            results[config_key]['evaluator_score'].sort(key=lambda x: x[0])
    
    return results


def get_individual_reasonableness_results(experiment_dir: str) -> Dict[str, Dict]:
    """
    Extract individual reasonableness evaluation results (not aggregated) for correctness filtering.

    Args:
        experiment_dir: Path to an experiment directory

    Returns:
        Dictionary with structure: {config_key: {iteration: [individual_results], 'path_info': {...}}}
    """
    results = {}
    exp_path = Path(experiment_dir)
    exp_name = exp_path.name

    # Walk through the directory structure to find simple_reasonable_recommendation evaluation JSON files
    for root, dirs, files in os.walk(experiment_dir):
        # Only process directories that contain simple_reasonable_recommendation
        if 'simple_reasonable_recommendation' not in root:
            continue

        # Filter for evaluation JSON files
        eval_files = [f for f in files if f.startswith('eval_') and f.endswith('.json')]

        if eval_files:
            # Sort by filename (which contains timestamps) and take the most recent one
            eval_files.sort(reverse=True)  # Sort in descending order to get most recent first
            file = eval_files[0]  # Take the most recent file
            file_path = os.path.join(root, file)

            # Extract path components to identify the experiment configuration
            rel_path = os.path.relpath(root, experiment_dir)
            path_parts = rel_path.split(os.sep)

            # Expected structure: inference_prompt_dir/eval_target/evaluator/eval_prompt_dir/iteration-X
            if len(path_parts) >= 5:
                inference_prompt = path_parts[0]
                eval_target = path_parts[1]
                evaluator = path_parts[2]
                eval_prompt = path_parts[3]
                iteration_dir = path_parts[4]

                # Extract iteration number from "iteration-X" or "iteration-base"
                if iteration_dir.startswith('iteration-'):
                    iteration_str = iteration_dir.replace('iteration-', '')
                    if iteration_str == 'base':
                        iteration = -1  # Use -1 for base model
                    else:
                        try:
                            iteration = int(iteration_str)
                        except ValueError:
                            continue
                else:
                    continue

                # Create a unique key for this configuration
                config_key = f"{inference_prompt}|{eval_target}|{evaluator}"

                try:
                    # Load the evaluation results
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Store individual results for this iteration
                    if config_key not in results:
                        results[config_key] = {
                            'path_info': {
                                'inference_prompt': inference_prompt,
                                'eval_target': eval_target,
                                'evaluator': evaluator,
                                'eval_prompt': eval_prompt
                            }
                        }

                    results[config_key][iteration] = data

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Error processing {file_path}: {e}")
                    continue

    return results


def filter_reasonableness_results_by_correctness(individual_results, evaluation_dir, prompt_type):
    """
    Filter individual reasonableness results into correct/incorrect/all categories.

    Args:
        individual_results (dict): Dictionary mapping config_key to {iteration: [individual_results]}
        evaluation_dir (str): Evaluation directory name
        prompt_type (str): Prompt type

    Returns:
        tuple: (all_results, correct_results, incorrect_results)
    """
    all_results = individual_results
    correct_results = {}
    incorrect_results = {}

    for config_key, config_data in individual_results.items():
        print(f"    Filtering results by correctness for config: {config_key}")

        correct_config = {}
        incorrect_config = {}

        # Copy path_info to new structures
        if 'path_info' in config_data:
            correct_config['path_info'] = config_data['path_info']
            incorrect_config['path_info'] = config_data['path_info']

        # Process each iteration
        for iteration, iteration_results in config_data.items():
            if iteration == 'path_info':
                continue

            correct_list = []
            incorrect_list = []
            unknown_list = []

            for result in iteration_results:
                example_index = result.get('example_index')
                if example_index is not None:
                    # Convert iteration to string format expected by correctness function
                    iteration_str = "base" if iteration == -1 else str(iteration)

                    correctness_info = get_correctness_for_datapoint(
                        evaluation_dir, prompt_type, None, iteration_str, example_index
                    )

                    if correctness_info['found'] and correctness_info['is_correct'] is not None:
                        if correctness_info['is_correct']:
                            correct_list.append(result)
                        else:
                            incorrect_list.append(result)
                    else:
                        unknown_list.append(result)
                else:
                    unknown_list.append(result)

            # Store results if we have any
            if correct_list:
                correct_config[iteration] = correct_list
            if incorrect_list:
                incorrect_config[iteration] = incorrect_list

            # Print counts for verification
            total_count = len(iteration_results)
            correct_count = len(correct_list)
            incorrect_count = len(incorrect_list)
            unknown_count = len(unknown_list)

            iteration_label = "base" if iteration == -1 else str(iteration + 1)
            print(f"      Iteration {iteration_label}: Total={total_count}, Correct={correct_count}, Incorrect={incorrect_count}, Unknown={unknown_count}")

        # Add to results if we have any data
        if any(k != 'path_info' for k in correct_config.keys()):
            correct_results[config_key] = correct_config
        if any(k != 'path_info' for k in incorrect_config.keys()):
            incorrect_results[config_key] = incorrect_config

    return all_results, correct_results, incorrect_results


def convert_individual_to_aggregated_results(individual_results):
    """
    Convert individual results back to the aggregated format expected by the plotting function.

    Args:
        individual_results: Dict with structure {config_key: {iteration: [individual_results]}}

    Returns:
        Dict with structure: {config_key: {'evaluator_score': [(iteration, mean_score, n_valid, n_total)]}}
    """
    aggregated_results = {}

    for config_key, config_data in individual_results.items():
        if not config_data:
            continue

        aggregated_config = {}

        # Copy path_info
        if 'path_info' in config_data:
            aggregated_config['path_info'] = config_data['path_info']

        scores_data = []

        # Process each iteration
        for iteration, iteration_results in config_data.items():
            if iteration == 'path_info':
                continue

            # Extract scores, excluding -1 (no answer) scores
            valid_scores = []
            total_examples = len(iteration_results)

            for item in iteration_results:
                if 'evaluator_score' in item and item['evaluator_score'] is not None:
                    score = item['evaluator_score']
                    if score != -1:  # Exclude "no answer" scores
                        valid_scores.append(score)

            if valid_scores:
                # Calculate mean score for this iteration (only from valid responses)
                mean_score = np.mean(valid_scores)
                n_valid = len(valid_scores)
                n_total = total_examples

                # Store (iteration, mean_score, n_valid, n_total)
                scores_data.append((iteration, mean_score, n_valid, n_total))

        if scores_data:
            # Sort by iteration
            scores_data.sort(key=lambda x: x[0])
            aggregated_config['evaluator_score'] = scores_data
            aggregated_results[config_key] = aggregated_config

    return aggregated_results


def create_reasonableness_plot(results, experiment_dir, output_dir, correctness_category="all"):
    """
    Create reasonableness plots for a given set of results and correctness category.

    Args:
        results: Dictionary with reasonableness results
        experiment_dir: Path to the experiment directory
        output_dir: Directory to save plots
        correctness_category: "all", "correct", or "incorrect"
    """
    exp_name = os.path.basename(experiment_dir)

    if not results:
        print(f"  No reasonableness evaluation data found for {exp_name} ({correctness_category})")
        return

    # Create base output directory structure that mirrors the evaluation structure
    eval_output_dir = "/nas/ucb/nikihowe/motivated-reasoning/evaluation_output"
    if experiment_dir.startswith(eval_output_dir):
        rel_path = os.path.relpath(experiment_dir, eval_output_dir)
        base_output_dir = os.path.join(output_dir, rel_path)
    else:
        base_output_dir = os.path.join(output_dir, exp_name)

    # Set up the plotting style (matching plot_simple_evaluation)
    plt.style.use('default')
    sns.set_palette("husl")

    # Plot each configuration
    for config_key, config_data in results.items():
        if 'evaluator_score' not in config_data:
            continue

        scores_data = config_data['evaluator_score']
        if not scores_data:
            continue

        # Get path information from the stored data
        if 'path_info' in config_data:
            path_info = config_data['path_info']
            inference_prompt = path_info['inference_prompt']
            eval_target = path_info['eval_target']
            evaluator = path_info['evaluator']
            eval_prompt = path_info['eval_prompt']

            # Create the full output path using original path components, including correctness category
            config_output_dir = os.path.join(base_output_dir, inference_prompt, eval_target, evaluator, eval_prompt, correctness_category)
            os.makedirs(config_output_dir, exist_ok=True)
        else:
            # Fallback to simple structure
            config_output_dir = os.path.join(base_output_dir, "reasonableness", correctness_category)
            os.makedirs(config_output_dir, exist_ok=True)

        # Separate iterations, scores, and counts
        if len(scores_data[0]) == 4:  # New format: (iteration, mean_score, n_valid, n_total)
            iterations, scores, n_valid_list, n_total_list = zip(*scores_data)
        else:  # Old format: (iteration, mean_score)
            iterations, scores = zip(*scores_data)
            n_valid_list = [None] * len(scores)
            n_total_list = [None] * len(scores)

        iterations = list(iterations)
        scores = list(scores)
        n_valid_list = list(n_valid_list)
        n_total_list = list(n_total_list)

        # Create the plot with same style as plot_simple_evaluation
        fig, ax = plt.subplots(figsize=(3.5, 3))

        # Force serif font for this figure (matching plot_simple_evaluation style)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

        # Plot the line
        ax.plot(range(len(iterations)), scores, marker='o', linewidth=2, markersize=6,
                label=f'{config_key}', alpha=0.8)

        # Add valid response counts above each point
        for i, (iteration, score, n_valid, n_total) in enumerate(zip(iterations, scores, n_valid_list, n_total_list)):
            if n_valid is not None:
                # Show valid count above the point (using i for x position since we use range(len(iterations)))
                ax.annotate(f'{n_valid}', (i, score),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=7, alpha=0.8)

        # Customize the plot (matching plot_simple_evaluation style)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reasonableness Score')

        # Create title using path info and correctness category
        if 'path_info' in config_data:
            path_info = config_data['path_info']
            human_preference = nice_format(exp_name)
            constitution = nice_format(path_info['inference_prompt'])

            # Add correctness category to title
            if correctness_category == "all":
                category_label = "Any"
            elif correctness_category == "correct":
                category_label = "Agree with Constitution"
            elif correctness_category == "incorrect":
                category_label = "Disagree with Constitution"
            else:
                category_label = correctness_category.title()

            ax.set_title(f'Preferences: {human_preference}\nConstitution: {constitution}\nResponses: {category_label}')
        else:
            ax.set_title(f'Reasonableness Scores Over Iterations\n{exp_name}\nResponses: {correctness_category.title()}')

        ax.grid(True, alpha=0.3)

        # Set y-axis range to 0-1 for reasonableness scores
        ax.set_ylim(0, 1)

        # Set x-axis labels (matching plot_simple_evaluation style)
        iterations_display = [f"base" if i == -1 else str(i + 1) for i in iterations]
        ax.set_xticks(range(len(iterations_display)))
        ax.set_xticklabels(iterations_display, fontsize=6)

        # Save the plot
        plot_filename = "reasonableness_score_over_iterations.png"
        plot_path = os.path.join(config_output_dir, plot_filename)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

        print(f"  Saved plot: {plot_path}")
        print(f"    Config: {config_key}")
        print(f"    Category: {correctness_category}")
        print(f"    Iterations: {iterations}")
        print(f"    Scores: {[f'{s:.3f}' for s in scores]}")
        if n_valid_list[0] is not None:
            print(f"    Valid responses: {n_valid_list}")
            print(f"    Total responses: {n_total_list}")


def plot_reasonableness_by_experiment(experiment_dir: str, output_dir: str) -> None:
    """
    Plot reasonableness scores for a specific experiment.

    Args:
        experiment_dir: Path to the experiment directory
        output_dir: Directory to save plots
    """
    exp_name = os.path.basename(experiment_dir)
    print(f"Processing experiment: {exp_name}")

    # Get individual results for correctness filtering
    individual_results = get_individual_reasonableness_results(experiment_dir)

    if not individual_results:
        print(f"  No reasonableness evaluation data found for {exp_name}")
        return

    # Determine the prompt type for correctness lookup
    # Extract from the first config's path_info
    prompt_type = None
    for config_data in individual_results.values():
        if 'path_info' in config_data:
            prompt_type = config_data['path_info']['inference_prompt']
            break

    if not prompt_type:
        print(f"  Could not determine prompt type for {exp_name}")
        return

    print(f"  Using prompt type: {prompt_type}")

    # Filter results by correctness
    all_individual_results, correct_individual_results, incorrect_individual_results = filter_reasonableness_results_by_correctness(
        individual_results, exp_name, prompt_type
    )

    # Convert back to aggregated format for plotting
    all_results = convert_individual_to_aggregated_results(all_individual_results)
    correct_results = convert_individual_to_aggregated_results(correct_individual_results)
    incorrect_results = convert_individual_to_aggregated_results(incorrect_individual_results)

    # Create plots for each correctness category
    categories = [
        ("all", all_results),
        ("correct", correct_results),
        ("incorrect", incorrect_results)
    ]

    for category_name, category_results in categories:
        if category_results:
            print(f"\n  Creating {category_name} plots...")
            create_reasonableness_plot(category_results, experiment_dir, output_dir, category_name)
        else:
            print(f"  No data for {category_name} category")


def plot_reasonableness(evaluation_output_dir: str, output_dir: str = "plots") -> None:
    """
    Main function to plot reasonableness scores for all experiments.
    
    Args:
        evaluation_output_dir: Path to the evaluation_output directory
        output_dir: Directory to save plots
    """
    print(f"Looking for experiments with reasonableness evaluations in: {evaluation_output_dir}")
    
    # Find all experiment directories with reasonableness evaluations
    experiment_dirs = find_experiment_directories(evaluation_output_dir)
    print(f"Found {len(experiment_dirs)} experiment directories with reasonableness evaluations")
    
    if not experiment_dirs:
        print("No experiments with reasonableness evaluations found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each experiment directory
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"\nProcessing experiment: {exp_name}")
        plot_reasonableness_by_experiment(exp_dir, output_dir)
    
    print(f"\nAll reasonableness plots saved to: {output_dir}")


def plot_experiment_by_name(experiment_name: str, evaluation_output_dir: str, output_dir: str) -> None:
    """
    Plot reasonableness scores for a specific experiment by name.
    
    Args:
        experiment_name: Name of the experiment to plot
        evaluation_output_dir: Path to the evaluation_output directory
        output_dir: Directory to save plots
    """
    experiment_path = os.path.join(evaluation_output_dir, experiment_name)
    
    if not os.path.exists(experiment_path):
        print(f"Error: Experiment directory {experiment_path} does not exist")
        print("Available experiments with reasonableness evaluations:")
        try:
            experiment_dirs = find_experiment_directories(evaluation_output_dir)
            for exp_dir in experiment_dirs:
                exp_name = os.path.basename(exp_dir)
                print(f"  - {exp_name}")
        except Exception as e:
            print(f"  Error listing experiments: {e}")
        return
    
    print(f"Plotting reasonableness for experiment: {experiment_name}")
    plot_reasonableness_by_experiment(experiment_path, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot reasonableness evaluation scores over iterations')
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
                       help='List available experiments with reasonableness evaluations')
    
    args = parser.parse_args()
    
    if args.list:
        # List available experiments
        experiment_dirs = find_experiment_directories(args.evaluation_output_dir)
        print("Available experiments with reasonableness evaluations:")
        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            print(f"  - {exp_name}")
    elif args.experiment:
        # Plot specific experiment
        plot_experiment_by_name(args.experiment, args.evaluation_output_dir, args.output_dir)
    else:
        # Plot all experiments (default)
        plot_reasonableness(args.evaluation_output_dir, args.output_dir)
