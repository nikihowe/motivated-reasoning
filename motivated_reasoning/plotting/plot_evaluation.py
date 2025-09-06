"""
Distribution plotting script for model evaluation structures.

This script creates distribution plots showing the distribution of scores across iterations.
"""

import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from collections import defaultdict
import argparse
import ast
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')  # Suppress matplotlib warnings in multiprocessing

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def infer_inference_script_name(prompt_type):
    """
    Infer the inference script name from the prompt type directory name.
    Args:
        prompt_type (str): The prompt type directory name (e.g., "simple_cot", "constitutional_cot")
    Returns:
        str: Human-readable inference script name
    """
    # Mapping from directory names to human-readable names
    inference_script_mapping = {
        "simple_cot": "Simple CoT",
        "constitutional_cot": "Constitutional CoT",
    }
    
    # Return mapped name if exists, otherwise use the directory name as-is
    return inference_script_mapping.get(prompt_type, prompt_type)


def infer_model_name(evaluation_dir):
    """
    Infer the model name from the evaluation directory name.
    Args:
        evaluation_dir (str): The evaluation directory name (e.g., "harmbench-08_28_213159", "harmbench_cot_tags-08_18_162853")
    Returns:
        str: Human-readable model name
    """
    # Mapping from directory patterns to human-readable names
    if evaluation_dir.startswith("harmbench_cot_tags"):
        return "CoT"
    elif evaluation_dir.startswith("hb_cot_const"):
        return "CoT + Constitution"
    elif evaluation_dir.startswith("harmbench"):
        return "No CoT"
    else:
        # For any other directory name, use as-is
        return evaluation_dir


def find_all_evaluators(evaluation_dir):
    """
    Find all evaluator directories in the evaluation output.
    Args:
        evaluation_dir (str): Name of the subfolder in evaluation_output to search
    Returns:
        list: List of evaluator names (without the "evaluator-" prefix)
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Evaluation directory {evaluation_path} does not exist")
        return []
    
    # Look for evaluator directories within prompt_type subdirectories
    evaluator_names = set()
    
    for prompt_type_dir in evaluation_path.iterdir():
        if prompt_type_dir.is_dir():
            evaluator_dirs = [d for d in prompt_type_dir.iterdir() 
                            if d.is_dir() and d.name.startswith("evaluator-")]
            for evaluator_dir in evaluator_dirs:
                evaluator_name = evaluator_dir.name.replace("evaluator-", "")
                evaluator_names.add(evaluator_name)
    
    return sorted(list(evaluator_names))


def load_evaluation_results_by_suffix(evaluation_dir, evaluator_name="base", prompt_type="cot_prompt"):
    """
    Load evaluation results organized by suffix condition.
    Args:
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator to load
        prompt_type (str): Prompt type to load (e.g., "training_prompt", "cot_prompt")
    Returns:
        dict: Dictionary mapping suffix names to results by iteration
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir / prompt_type / f"evaluator-{evaluator_name}"
    
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
            # Extract iteration number from directory name
            iteration = int(iteration_dir.name.split("-")[1])
            
            # Look for eval JSON files in this iteration
            json_files = list(iteration_dir.glob("*eval*.json"))
            if not json_files:
                continue
                
            # Load the most recent JSON file (by modification time)
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            print(f"Loading evaluation results from: {latest_file}")
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


def analyze_evaluation_results(results_by_iteration, score_key):
    """
    Analyze evaluation results and compute summary statistics.
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        score_key (str): Which score to analyze
    Returns:
        dict: Summary statistics for each iteration
    """
    summary_stats = {}
    
    for iteration, results in results_by_iteration.items():
        all_scores = []
        score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, -1: 0}  # -1 for no score
        
        for result in results:
            score = result.get(score_key)
            if score is not None and score != -1:
                all_scores.append(score)
                score_distribution[score] = score_distribution.get(score, 0) + 1
            else:
                score_distribution[-1] += 1
        
        if all_scores:
            summary_stats[iteration] = {
                'all_scores': all_scores,
                'mean': np.mean(all_scores),
                'median': np.median(all_scores),
                'std': np.std(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores),
                'count': len(all_scores),
                'score_distribution': {}
            }
            
            # Calculate distribution percentages
            total = len(all_scores)
            for score, count in score_distribution.items():
                if score != -1:  # Skip no score for percentage calculation
                    percentage = (count / total) * 100
                    summary_stats[iteration]['score_distribution'][score] = {
                        'count': count,
                        'percentage': percentage
                    }
    
    return summary_stats


def create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name="base", prompt_type="cot_prompt", use_argmax=True):
    """
    Create a stacked bar chart showing the distribution of scores across iterations.
    """
    iterations = sorted(results_by_iteration.keys())
    
    if not iterations:
        print(f"No data to plot for {label} - {suffix_name}!")
        return
    
    # Prepare data for plotting
    scores_1 = []
    scores_2 = []
    scores_3 = []
    scores_4 = []
    scores_5 = []
    scores_neg1 = []
    
    for iteration in iterations:
        results = results_by_iteration[iteration]
        
        # Count scores for this iteration
        score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, -1: 0}
        
        for result in results:
            score = result.get(score_key)
            if score is not None and score in score_counts:
                score_counts[score] += 1
            else:
                score_counts[-1] += 1
        
        scores_1.append(score_counts[1])
        scores_2.append(score_counts[2])
        scores_3.append(score_counts[3])
        scores_4.append(score_counts[4])
        scores_5.append(score_counts[5])
        scores_neg1.append(score_counts[-1])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.6
    
    # Use custom colors for training prompt evaluations
    if score_key == "classification_score":
        # Training prompt colors: -1=grey, 1=blue, 2=light blue, 3=red, 4=light red, 5=black
        # Even more pastel/softer versions
        colors = ['#969696', '#9ecae1', '#c6dbef', '#e6550d', '#fdbe85', '#969696']
    else:
        # Default colors for other score types: -1=grey, 1=blue, 2=green, 3=orange, 4=red, 5=purple
        colors = ['#969696', '#1f77b4', '#2ca02c', '#ffbb78', '#ff7f0e', '#d62728']
    
    # Create legend labels based on score type (Score 5 at top, No Score at bottom)
    if score_key == "classification_score":
        # Training prompt legend labels
        labels = ['5: No recommendation', '4: Second option with made up reason', '3: Second option', '2: First option with made up reason', '1: First option', 'No Score']
    else:
        # Default labels for other score types
        labels = ['Score 5', 'Score 4', 'Score 3', 'Score 2', 'Score 1', 'No Score']
    
    # Create bars in stacking order (No Score at bottom, Score 5 at top) with matching legend order
    p_neg1 = ax.bar(iterations, scores_neg1, width, label=labels[5], color=colors[0])
    p1 = ax.bar(iterations, scores_1, width, bottom=scores_neg1, label=labels[4], color=colors[1])
    p2 = ax.bar(iterations, scores_2, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1), 
                label=labels[3], color=colors[2])
    p3 = ax.bar(iterations, scores_3, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1) + np.array(scores_2), 
                label=labels[2], color=colors[3])
    p4 = ax.bar(iterations, scores_4, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1) + np.array(scores_2) + np.array(scores_3), 
                label=labels[1], color=colors[4])
    p5 = ax.bar(iterations, scores_5, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1) + np.array(scores_2) + np.array(scores_3) + np.array(scores_4), 
                label=labels[0], color=colors[5])
    
    method_name = "Argmax" if use_argmax else "Weighted Average"
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Number of Examples', fontsize=14)
    # Create custom title with model name and inference script
    # Always infer model name and inference script from directory structure
    model_name = infer_model_name(evaluation_dir)
    inference_script = infer_inference_script_name(prompt_type)
    title = f'Training method: {model_name}    Inference Prompt: {inference_script}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis ticks to show every iteration
    ax.set_xticks(iterations)
    ax.set_xticklabels(iterations)
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Create legend with explicit order (Score 5 at top, No Score at bottom)
    handles = [p5, p4, p3, p2, p1, p_neg1]
    legend_labels = ['Fully motivated reasoning', 'Mostly motivated reasoning', 'Mixed Reasoning', 'Mostly genuine reasoning', 'Fully genuine reasoning', 'Evaluator returned no score']
    
    legend = ax.legend(
        handles, legend_labels,
        framealpha=0.95,           # 95% opacity
        facecolor='white',          # White background
        edgecolor='black',          # Black border for better contrast
        fancybox=True,              # Rounded corners
        shadow=True,                # Add shadow
        fontsize=10,                # Slightly smaller font
        bbox_to_anchor=(1.02, 1),  # Position legend outside plot area
        loc='upper left'            # Upper left position
    )
    
    # Ensure the legend frame is visible
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.95)
    
    ax.grid(True, axis='y', alpha=0.3)
    
    # Save the plot
    if score_key == "full_influence_score":
        score_type_dir = "whole_response"
    elif score_key == "reasoning_influence_score":
        score_type_dir = "reasoning_only"
    elif score_key == "classification_score":
        score_type_dir = "classification"
    else:
        score_type_dir = "other"
    
    method_subdir = "argmax" if use_argmax else "weighted_avg"
    plots_dir = Path("plots") / evaluation_dir / prompt_type / f"evaluator-{evaluator_name}" / score_type_dir / "eval" / suffix_name / "distribution" / method_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "score_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to: {plot_path}")
    plt.close()


def process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name, prompt_type="cot_prompt"):
    """
    Process a single suffix condition and create distribution plots.
    Args:
        suffix_name (str): Name of the suffix condition
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator
        prompt_type (str): Optional prompt type to determine which scores to plot
    """
    print(f"  Processing suffix: {suffix_name}")
    
    if not results_by_iteration:
        print(f"    No data found for suffix: {suffix_name}")
        return
    
    # Determine which scores to process based on evaluator type
    score_keys_and_labels = [
        ("full_evaluation_score", "Full Response Evaluation"),
        ("reasoning_evaluation_score", "Reasoning Only Evaluation")
    ]
    
    # Process each score type
    for score_key, label in score_keys_and_labels:
        print(f"    Creating distribution plots for {label}...")
        
        # Create argmax version
        print(f"      Creating argmax distribution plot...")
        create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name, prompt_type, use_argmax=True)
        
        # Create weighted average version  
        print(f"      Creating weighted average distribution plot...")
        create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name, prompt_type, use_argmax=False)
    
    plt.close('all')  # Close all plots to free memory
    
    print(f"    Completed all distribution plots for {suffix_name}")
    return f"Completed all distribution plots for {suffix_name}"


def process_evaluator(evaluation_dir, evaluator_name, suffix_filter=None, prompt_type=None):
    """
    Process results for a single evaluator and create distribution plots.
    Args:
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator to process
        suffix_filter (str): Optional suffix to filter to
        prompt_type (str): Optional prompt type to filter by (e.g., "training_prompt", "cot_prompt")
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING EVALUATOR: {evaluator_name}")
    print(f"{'='*80}")
    
    # If no prompt_type specified, find all available prompt types for this evaluator
    if not prompt_type:
        evaluation_path = Path("evaluation_output") / evaluation_dir
        available_prompt_types = []
        for prompt_dir in evaluation_path.iterdir():
            if prompt_dir.is_dir():
                evaluator_path = prompt_dir / f"evaluator-{evaluator_name}"
                if evaluator_path.exists():
                    available_prompt_types.append(prompt_dir.name)
        
        if not available_prompt_types:
            print(f"No prompt types found for evaluator-{evaluator_name}")
            return
        
        print(f"Found prompt types: {available_prompt_types}")
        prompt_types_to_process = available_prompt_types
    else:
        prompt_types_to_process = [prompt_type]
    
    # Process each prompt type
    for pt in prompt_types_to_process:
        print(f"\nProcessing prompt type: {pt}")
        
        # Load results for this evaluator and prompt type
        results_by_suffix = load_evaluation_results_by_suffix(evaluation_dir, evaluator_name, pt)
        
        if not results_by_suffix:
            print(f"  No data found for evaluator-{evaluator_name} with prompt type {pt}")
            continue
        
        print(f"  Found {len(results_by_suffix)} suffix conditions: {list(results_by_suffix.keys())}")
        
        # Process each suffix condition
        for suffix_name, results_by_iteration in results_by_suffix.items():
            if suffix_filter and suffix_name != suffix_filter:
                continue
                
            print(f"  Processing suffix: {suffix_name}")
            process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name, pt)
        
        print(f"\nCompleted processing evaluator-{evaluator_name} with prompt type {pt}!")
    
    print(f"\nCompleted processing evaluator-{evaluator_name}!")


def main():
    parser = argparse.ArgumentParser(description='Create distribution plots for model evaluation results')
    parser.add_argument('evaluation_dir', type=str, help='Evaluation directory name')
    parser.add_argument('--suffix', type=str, help='Specific suffix condition to analyze (optional)')
    parser.add_argument('--evaluator', type=str, help='Specific evaluator name (default: process all evaluators)')
    parser.add_argument('--prompt-type', type=str, help='Prompt type to analyze (e.g., "training_prompt", "cot_prompt")')
    parser.add_argument('--list-evaluators', action='store_true', help='List available evaluators and exit')
    args = parser.parse_args()
    
    evaluation_dir = args.evaluation_dir
    
    # If no prompt type specified, find all available prompt types
    if args.prompt_type:
        prompt_type = args.prompt_type
        print(f"Loading evaluation results from: {evaluation_dir}")
        print(f"Using prompt type: {prompt_type}")
    else:
        # Auto-detect available prompt types
        evaluation_path = Path("evaluation_output") / evaluation_dir
        if not evaluation_path.exists():
            print(f"Error: Evaluation directory {evaluation_path} does not exist")
            return
        
        available_prompt_types = []
        for prompt_dir in evaluation_path.iterdir():
            if prompt_dir.is_dir():
                available_prompt_types.append(prompt_dir.name)
        
        if not available_prompt_types:
            print(f"Error: No prompt type directories found in {evaluation_path}")
            return
        
        print(f"Loading evaluation results from: {evaluation_dir}")
        print(f"Found {len(available_prompt_types)} prompt type(s): {available_prompt_types}")
        print(f"Processing all prompt types automatically")
    
    # Find all available evaluators
    available_evaluators = find_all_evaluators(evaluation_dir)
    
    if not available_evaluators:
        print("No evaluator directories found!")
        return
    
    print(f"Found {len(available_evaluators)} evaluator(s): {available_evaluators}")
    
    # List evaluators and exit if requested
    if args.list_evaluators:
        print("\nAvailable evaluators:")
        for evaluator in available_evaluators:
            print(f"  - evaluator-{evaluator}")
        return
    
    # Determine which evaluators to process
    if args.evaluator:
        # Process specific evaluator
        if args.evaluator not in available_evaluators:
            print(f"Evaluator '{args.evaluator}' not found. Available evaluators: {available_evaluators}")
            return
        evaluators_to_process = [args.evaluator]
    else:
        # Process all evaluators
        evaluators_to_process = available_evaluators
    
    print(f"\nProcessing {len(evaluators_to_process)} evaluator(s): {evaluators_to_process}")
    
    # Process all prompt types if none was specified
    if args.prompt_type:
        # Single prompt type specified
        prompt_types_to_process = [prompt_type]
    else:
        # Auto-detected prompt types
        prompt_types_to_process = available_prompt_types
    
    print(f"Processing {len(prompt_types_to_process)} prompt type(s): {prompt_types_to_process}")
    
    # Process each prompt type
    for pt in prompt_types_to_process:
        print(f"\n{'='*80}")
        print(f"PROCESSING PROMPT TYPE: {pt}")
        print(f"{'='*80}")
        
        # Process evaluators sequentially
        print(f"Processing evaluators sequentially...")
        for i, evaluator_name in enumerate(evaluators_to_process):
            print(f"\n[{i+1}/{len(evaluators_to_process)}] About to process evaluator: {evaluator_name}")
            process_evaluator(evaluation_dir, evaluator_name, args.suffix, pt)
            print(f"[{i+1}/{len(evaluators_to_process)}] Completed processing evaluator: {evaluator_name}")
    
    print(f"\n{'='*80}")
    print("ALL DISTRIBUTION PLOTS COMPLETED!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()