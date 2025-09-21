import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns


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
    
    Args:
        experiment_dir: Path to an experiment directory
        evaluator_only: If True, only include experiments with evaluator-based structure
        
    Returns:
        Dictionary with structure: {experiment_name: {score_type: [(iteration, score), ...]}}
    """
    results = {}
    exp_path = Path(experiment_dir)
    
    # Walk through the directory structure to find evaluation JSON files
    for root, dirs, files in os.walk(experiment_dir):
        # Filter for evaluation JSON files
        eval_files = [f for f in files if f.startswith('eval_') and f.endswith('.json')]
        
        if eval_files:
            # Sort by filename (which contains timestamps) and take the most recent one
            eval_files.sort(reverse=True)  # Sort in descending order to get most recent first
            file = eval_files[0]  # Take the most recent file
            file_path = os.path.join(root, file)
            
            # Extract experiment path components
            rel_path = os.path.relpath(root, experiment_dir)
            path_parts = rel_path.split(os.sep)
            
            # Check if this is an evaluator-based experiment
            if evaluator_only:
                # Look for evaluator in the path (e.g., evaluator-gemini-25-flash-lite)
                has_evaluator = any('evaluator-' in part for part in path_parts)
                if not has_evaluator:
                    continue
            
            # Create a meaningful experiment name from the path
            if len(path_parts) >= 2:
                experiment_name = '/'.join(path_parts[:-1])  # Exclude the iteration directory
                iteration_dir = path_parts[-1]
                
                # Extract iteration number
                if iteration_dir.startswith('iteration-'):
                    try:
                        if iteration_dir == 'iteration-base':
                            iteration_num = -1  # Use -1 for base iteration
                        else:
                            iteration_num = int(iteration_dir.split('-')[1])
                    except (ValueError, IndexError):
                        continue
                else:
                    continue
                
                # Load and parse the JSON file
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract scores from the data
                    if isinstance(data, list) and len(data) > 0:
                        # Calculate average scores across all examples
                        score_types = ['response_evaluation_score', 'reasoning_evaluation_score', 'full_evaluation_score']
                        
                        for score_type in score_types:
                            scores = []
                            for item in data:
                                if score_type in item and item[score_type] is not None:
                                    scores.append(item[score_type])
                            
                            if scores:  # Only add if we have valid scores
                                avg_score = np.mean(scores)
                                
                                # Initialize nested dictionaries if needed
                                if experiment_name not in results:
                                    results[experiment_name] = {}
                                if score_type not in results[experiment_name]:
                                    results[experiment_name][score_type] = []
                                
                                results[experiment_name][score_type].append((iteration_num, avg_score))
                
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    # Sort iterations for each experiment and score type
    for exp_name in results:
        for score_type in results[exp_name]:
            results[exp_name][score_type].sort(key=lambda x: x[0])
    
    return results


def plot_evaluation_scores(results: Dict[str, Dict[str, List[Tuple[int, float]]]], 
                          experiment_dir_name: str, 
                          output_dir: str = "plots",
                          experiment_path: str = None) -> None:
    """
    Plot evaluation scores over iterations for all experiments.
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
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Group results by model type (more specific grouping)
    subdir_groups = {}
    for exp_name, exp_data in results.items():
        # Split the experiment name to get a more meaningful grouping
        path_parts = exp_name.split('/')
        if len(path_parts) >= 2:
            # Use first two parts for more specific grouping
            # e.g., 'risky_constitutional_cot/reasoning' vs 'risky_constitutional_cot/response'
            subdir = '/'.join(path_parts[:2])  # e.g., 'risky_constitutional_cot/reasoning'
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
        
        # Get all unique score types for this subdirectory
        all_score_types = set()
        for exp_data in subdir_results.values():
            all_score_types.update(exp_data.keys())
        
        # Create plots for each score type in this subdirectory
        for score_type in all_score_types:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each experiment in this subdirectory
            for exp_name, exp_data in subdir_results.items():
                if score_type in exp_data and exp_data[score_type]:
                    iterations, scores = zip(*exp_data[score_type])
                    
                    # Convert base iteration (-1) to a more readable label
                    iterations_display = [f"base" if i == -1 else str(i) for i in iterations]
                    
                    ax.plot(range(len(iterations)), scores, marker='o', linewidth=2, 
                           markersize=6, label=exp_name, alpha=0.8)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel(f'{score_type.replace("_", " ").title()}', fontsize=12)
            ax.set_title(f'{score_type.replace("_", " ").title()} Across Iterations\n{experiment_dir_name}/{subdir}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set y-axis range for specific score types
            if score_type == 'reasoning_evaluation_score':
                ax.set_ylim(1, 5)
            elif score_type == 'response_evaluation_score':
                ax.set_ylim(0, 1)
            
            # Set x-axis labels
            if subdir_results:  # If we have any results
                sample_exp = next(iter(subdir_results.values()))
                if score_type in sample_exp:
                    sample_iterations, _ = zip(*sample_exp[score_type])
                    iterations_display = [f"base" if i == -1 else str(i) for i in sample_iterations]
                    ax.set_xticks(range(len(iterations_display)))
                    ax.set_xticklabels(iterations_display)
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"{score_type}_over_iterations.png"
            filepath = os.path.join(subdir_output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
            
            plt.show()
            plt.close()


def plot_simple_evaluation(evaluation_output_dir: str, output_dir: str = "plots") -> None:
    """
    Main function to plot evaluation scores for all experiments in the evaluation_output directory.
    
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
        print(f"\nProcessing experiment: {exp_name}")
        
        # Extract evaluation scores (evaluator-based only)
        results = extract_evaluation_scores(exp_dir, evaluator_only=True)
        
        if results:
            print(f"Found {len(results)} sub-experiments with evaluation data")
            # Create plots with full path for structure preservation
            plot_evaluation_scores(results, exp_name, output_dir, exp_dir)
        else:
            print(f"No evaluation data found in {exp_dir}")


def plot_specific_experiment(experiment_dir: str, output_dir: str = "plots") -> None:
    """
    Plot evaluation scores for a specific experiment directory.
    
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
    
    # Extract evaluation scores (evaluator-based only)
    results = extract_evaluation_scores(experiment_dir, evaluator_only=True)
    
    if results:
        print(f"Found {len(results)} sub-experiments with evaluation data")
        # Create plots with directory structure preservation
        plot_evaluation_scores(results, exp_name, output_dir, experiment_dir)
    else:
        print(f"No evaluation data found in {experiment_dir}")


def plot_experiment_by_name(experiment_name: str, evaluation_output_dir: str = "/nas/ucb/nikihowe/motivated-reasoning/evaluation_output", output_dir: str = "plots") -> None:
    """
    Plot evaluation scores for a specific experiment by name.
    
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
    
    parser = argparse.ArgumentParser(description='Plot evaluation scores over iterations')
    parser.add_argument('--experiment', '-e', type=str, 
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
        plot_simple_evaluation(args.evaluation_output_dir, args.output_dir)
