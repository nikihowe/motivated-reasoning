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
from motivated_reasoning.plotting.plotting_utils import nice_format


# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
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
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2
})


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
        return "Constitutional CoT"
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
    
    # Look for evaluator directories within prompt_type/eval_target subdirectories
    evaluator_names = set()
    
    for prompt_type_dir in evaluation_path.iterdir():
        if prompt_type_dir.is_dir():
            # Look for eval_target subdirectories (full, reasoning, response)
            for eval_target_dir in prompt_type_dir.iterdir():
                if eval_target_dir.is_dir():
                    evaluator_dirs = [d for d in eval_target_dir.iterdir() 
                                    if d.is_dir() and d.name.startswith("evaluator-")]
                    for evaluator_dir in evaluator_dirs:
                        evaluator_name = evaluator_dir.name.replace("evaluator-", "")
                        evaluator_names.add(evaluator_name)
    
    return sorted(list(evaluator_names))


def load_evaluation_results_by_suffix(evaluation_dir, evaluator_name="base", prompt_type="cot_prompt", eval_target="full"):
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
            
            # print(f"Loading evaluation results from: {latest_file}")
            try:
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                results_by_iteration[iteration] = results
                # print(f"  Loaded {len(results)} evaluation examples for {suffix_name} ({prompt_type}) iteration {iteration}")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
                continue
        
        if results_by_iteration:
            results_by_suffix[suffix_name] = results_by_iteration
    
    return results_by_suffix


def get_correctness_for_datapoint(evaluation_dir, prompt_type, evaluator_name, iteration, example_index):
    """
    Get the correctness (response_evaluation_score) for a specific datapoint.
    Always uses flash-lite evaluator for correctness judgments regardless of which evaluator
    is being used for other evaluations.
    
    Args:
        evaluation_dir (str): Evaluation directory name
        prompt_type (str): Prompt type (e.g., "safe_constitutional_cot")
        evaluator_name (str): Evaluator name (ignored - always uses flash-lite)
        iteration (str): Iteration identifier
        example_index (int): Index of the example to look up
    
    Returns:
        dict: Contains 'is_correct' (bool), 'score' (int), and 'found' (bool)
    """
    # Always use flash-lite evaluator for correctness data
    correctness_path = Path("evaluation_output") / evaluation_dir / prompt_type / "response" / "evaluator-gemini-25-flash-lite" / "simple_correct_choice" / f"iteration-{iteration}"
    
    if not correctness_path.exists():
        return {'found': False, 'is_correct': None, 'score': None}
    
    # Find the most recent JSON file
    json_files = list(correctness_path.glob("*eval*.json"))
    if not json_files:
        return {'found': False, 'is_correct': None, 'score': None}
    
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        # Find the entry with matching example_index
        for result in results:
            if result.get('example_index') == example_index:
                response_score = result.get('evaluator_score')
                # Determine correctness based on evaluator_score (since this is in response directory)
                # Score of 1 = correct/genuine, higher scores = more motivated/incorrect
                is_correct = response_score == 1 if response_score is not None else None
                return {
                    'found': True,
                    'is_correct': is_correct,
                    'score': response_score
                }
        
        # Example index not found
        return {'found': False, 'is_correct': None, 'score': None}
        
    except Exception as e:
        print(f"Error reading correctness file {latest_file}: {e}")
        return {'found': False, 'is_correct': None, 'score': None}


def filter_results_by_correctness(results_by_iteration, evaluation_dir, prompt_type, evaluator_name, suffix_name):
    """
    Filter results into correct/incorrect/all categories.
    
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        evaluation_dir (str): Evaluation directory name
        prompt_type (str): Prompt type
        evaluator_name (str): Evaluator name
        suffix_name (str): Suffix name for logging
    
    Returns:
        tuple: (all_results, correct_results, incorrect_results)
    """
    all_results = results_by_iteration
    correct_results = {}
    incorrect_results = {}
    
    print(f"    Filtering results by correctness for suffix: {suffix_name}")
    
    # Sort iterations with custom logic: "base" first, then numeric iterations
    def sort_iterations(iteration_key):
        if iteration_key == "base":
            return -1  # "base" comes first
        else:
            try:
                return int(iteration_key)
            except ValueError:
                return float('inf')  # Unknown iterations go last

    sorted_iterations = sorted(results_by_iteration.keys(), key=sort_iterations)
    
    for iteration in sorted_iterations:
        results = results_by_iteration[iteration]
        correct_list = []
        incorrect_list = []
        unknown_list = []
        
        for result in results:
            example_index = result.get('example_index')
            if example_index is not None:
                correctness_info = get_correctness_for_datapoint(
                    evaluation_dir, prompt_type, evaluator_name, iteration, example_index
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
        
        # Store results
        if correct_list:
            correct_results[iteration] = correct_list
        if incorrect_list:
            incorrect_results[iteration] = incorrect_list
        
        # Print detailed counts for verification
        total_count = len(results)
        correct_count = len(correct_list)
        incorrect_count = len(incorrect_list)
        unknown_count = len(unknown_list)
        
        print(f"      Iteration {iteration}: Total={total_count}, Correct={correct_count}, Incorrect={incorrect_count}, Unknown={unknown_count}")
    
    return all_results, correct_results, incorrect_results


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


def create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name="base", prompt_type="cot_prompt", use_argmax=True, eval_target="full", correctness_category="all"):
    """
    Create a stacked bar chart showing the distribution of scores across iterations.
    """
    # Sort iterations with custom logic: "base" first, then numeric iterations
    def sort_iterations(iteration_key):
        if iteration_key == "base":
            return -1  # "base" comes first
        else:
            try:
                return int(iteration_key)
            except ValueError:
                return float('inf')  # Unknown iterations go last

    iterations = sorted(results_by_iteration.keys(), key=sort_iterations)

    # Truncate to only show base + iterations 0-10 (first 11 iterations)
    iterations = iterations[:11]

    # Create mapping from iteration names to x-coordinates
    # "base" -> 0, "0" -> 1, "1" -> 2, etc.
    x_positions = []
    x_labels = []
    for i, iteration in enumerate(iterations):
        if iteration == "base":
            x_positions.append(0)
            x_labels.append("base")
        else:
            try:
                numeric_iter = int(iteration)
                x_positions.append(numeric_iter + 1)  # Shift numeric iterations by 1
                x_labels.append(str(numeric_iter + 1))
            except ValueError:
                x_positions.append(i)  # Fallback for unknown iterations
                x_labels.append(iteration)
    
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
        
        # Convert counts to percentages
        total_count = sum(score_counts.values())
        if total_count > 0:
            scores_1.append((score_counts[1] / total_count) * 100)
            scores_2.append((score_counts[2] / total_count) * 100)
            scores_3.append((score_counts[3] / total_count) * 100)
            scores_4.append((score_counts[4] / total_count) * 100)
            scores_5.append((score_counts[5] / total_count) * 100)
            scores_neg1.append((score_counts[-1] / total_count) * 100)
        else:
            scores_1.append(0)
            scores_2.append(0)
            scores_3.append(0)
            scores_4.append(0)
            scores_5.append(0)
            scores_neg1.append(0)
    
    # Create the plot with publication-ready size (4 inches wide) and space for spectrum
    fig, (ax, cax) = plt.subplots(2, 1, figsize=(3.5, 3), 
                                  gridspec_kw={'height_ratios': [4, 0.4], 'hspace': 0.4})
    
    width = 0.9
    
    # Create a professional green-to-red spectrum with muted, sophisticated colors
    colors = {
        -1: '#8c8c8c',  # Professional grey for no score
        1: '#2d5a3d',   # Deep forest green (fully genuine)
        2: '#4a7c59',   # Muted green (mostly genuine) 
        3: '#d4af37',   # Sophisticated gold (mixed)
        4: '#b85450',   # Muted red-brown (mostly motivated)
        5: '#8b2635'    # Deep burgundy (fully motivated)
    }
    
    # Create bars in stacking order (No Score at bottom, Score 5 at top)
    p_neg1 = ax.bar(x_positions, scores_neg1, width, color=colors[-1])
    p1 = ax.bar(x_positions, scores_1, width, bottom=scores_neg1, color=colors[1])
    p2 = ax.bar(x_positions, scores_2, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1), 
                color=colors[2])
    p3 = ax.bar(x_positions, scores_3, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1) + np.array(scores_2), 
                color=colors[3])
    p4 = ax.bar(x_positions, scores_4, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1) + np.array(scores_2) + np.array(scores_3), 
                color=colors[4])
    p5 = ax.bar(x_positions, scores_5, width, 
                bottom=np.array(scores_neg1) + np.array(scores_1) + np.array(scores_2) + np.array(scores_3) + np.array(scores_4), 
                color=colors[5])
    
    method_name = "Argmax" if use_argmax else "Weighted Average"
    
    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel('Percentage of Examples')
    # Create custom title with correctness category
    model_name = infer_model_name(evaluation_dir)
    preference = nice_format(model_name)
    constitution = nice_format(prompt_type)
    
    # Add correctness category to title
    if correctness_category == "all":
        category_label = "Any"
    elif correctness_category == "correct":
        category_label = "Agree with Constitution"
    elif correctness_category == "incorrect":
        category_label = "Disagree with Constitution"
    else:
        category_label = correctness_category.title()
    
    title = f'Preferences: {preference}\nConstitution: {constitution}'#\nResponses: {category_label}'
    
    ax.set_title(title, fontsize=12)
    
    # Set x-axis ticks to show every iteration with custom labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    
    # Reduce horizontal whitespace by tightening x-axis limits
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.6, max(x_positions) + 0.6)
    
    # Set y-axis to show percentages from 0 to 100%, with extra space for count labels
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    # Add count labels above each bar
    for i, (x_pos, iteration) in enumerate(zip(x_positions, iterations)):
        results = results_by_iteration[iteration]
        count = len(results)
        # Position the text slightly above the 100% mark
        ax.text(x_pos, 102, f'{count}', ha='center', va='bottom', fontsize=6)
    
    # Clean grid styling (horizontal lines only, no vertical lines)
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5, axis='y')
    
    # Create color spectrum bar below the main plot using rectangles
    spectrum_width = 0.8
    spectrum_height = 0.4
    spectrum_y = 0.5
    
    for i, score in enumerate([1, 2, 3, 4, 5]):
        rect = plt.Rectangle((i * spectrum_width, spectrum_y), spectrum_width, spectrum_height, 
                           facecolor=colors[score], edgecolor='white', linewidth=0.8)
        cax.add_patch(rect)
    
    # Set axis limits and appearance (expanded for text and grey box)
    cax.set_xlim(-1.0, 6.0)
    cax.set_ylim(0, 1)
    cax.set_aspect('equal')
    
    # Remove all tick marks for a clean spectrum appearance
    cax.set_xticks([])
    cax.set_yticks([])
    
    # Add explanatory text positioned more to the left, with grey box for "No Score"
    cax.text(-2.3, 0.65, 'Fully Genuine', ha='center', va='center', fontsize=7, fontweight='bold')
    cax.text(6.5, 0.65, 'Fully Motivated', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Add small grey box with "No Score" label
    # grey_box = plt.Rectangle((spectrum_width, spectrum_y + 0.5), spectrum_width, spectrum_height, facecolor=colors[-1], edgecolor='white', linewidth=0.8)
    # cax.add_patch(grey_box)
    # cax.text(5.45, 0.5, 'No Score', ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Remove spectrum axis spines
    for spine in cax.spines.values():
        spine.set_visible(False)
    
    # Save the plot - use eval_target as the score_type_dir since that determines the evaluation type
    if eval_target == "full" or eval_target == "everything":
        score_type_dir = "whole_response"
    elif eval_target == "reasoning":
        score_type_dir = "reasoning_only"
    elif eval_target == "response":
        score_type_dir = "response_only"
    elif eval_target in ["system_and_reasoning", "constitution_and_reasoning"]:
        score_type_dir = "system_reasoning"
    elif eval_target in ["system_and_response", "constitution_and_response"]:
        score_type_dir = "system_response"
    elif eval_target in ["constitution_and_reasoning_and_response"]:
        score_type_dir = "constitution_full"
    else:
        # Use eval_target as-is for unknown types
        score_type_dir = eval_target.replace("_", "-")
    
    method_subdir = "argmax" if use_argmax else "weighted_avg"
    plots_dir = Path("plots") / evaluation_dir / prompt_type / eval_target / f"evaluator-{evaluator_name}" / score_type_dir / "eval" / suffix_name / "distribution" / method_subdir / correctness_category
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "score_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save as PDF for vector graphics
    pdf_path = plot_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nSaved distribution plot to: {plot_path}")
    plt.close()


def process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name, prompt_type="cot_prompt", eval_target="full"):
    """
    Process a single suffix condition and create distribution plots.
    Args:
        suffix_name (str): Name of the suffix condition
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator
        prompt_type (str): Optional prompt type to determine which scores to plot
        eval_target (str): Evaluation target (full, reasoning, response)
    """
    print(f"  Processing suffix: {suffix_name} (eval_target: {eval_target})")
    
    if not results_by_iteration:
        print(f"    No data found for suffix: {suffix_name}")
        return
    
    # Filter results by correctness
    all_results, correct_results, incorrect_results = filter_results_by_correctness(
        results_by_iteration, evaluation_dir, prompt_type, evaluator_name, suffix_name
    )
    
    # In the new format, all evaluations use 'evaluator_score' field
    # The evaluation type is determined by the directory structure (eval_target)
    if eval_target == "full" or eval_target == "everything":
        score_keys_and_labels = [("evaluator_score", "Full Response Evaluation")]
    elif eval_target == "reasoning":
        score_keys_and_labels = [("evaluator_score", "Reasoning Only Evaluation")]
    elif eval_target == "response":
        score_keys_and_labels = [("evaluator_score", "Response Evaluation")]
    elif eval_target in ["system_and_reasoning", "constitution_and_reasoning"]:
        score_keys_and_labels = [("evaluator_score", "System + Reasoning Evaluation")]
    elif eval_target in ["system_and_response", "constitution_and_response"]:
        score_keys_and_labels = [("evaluator_score", "System + Response Evaluation")]
    elif eval_target in ["constitution_and_reasoning_and_response"]:
        score_keys_and_labels = [("evaluator_score", "Constitution + Reasoning + Response Evaluation")]
    else:
        # Fallback for unknown eval_target
        score_keys_and_labels = [("evaluator_score", f"{eval_target.title()} Evaluation")]
    
    # Create plots for each correctness category
    correctness_categories = [
        ("all", all_results, "All Examples"),
        ("correct", correct_results, "Correct Examples Only"), 
        ("incorrect", incorrect_results, "Incorrect Examples Only")
    ]
    
    # Process each score type and correctness category
    for score_key, label in score_keys_and_labels:
        print(f"    Creating distribution plots for {label}...")
        
        for category_name, category_results, category_label in correctness_categories:
            if not category_results:
                print(f"      Skipping {category_label} - no data available")
                continue
                
            print(f"      Creating plots for {category_label}...")
            
            # Create argmax version
            print(f"        Creating argmax distribution plot...")
            create_score_distribution_plot(category_results, evaluation_dir, suffix_name, score_key, label, 
                                         evaluator_name, prompt_type, use_argmax=True, eval_target=eval_target, 
                                         correctness_category=category_name)
            
            # Create weighted average version  
            print(f"        Creating weighted average distribution plot...")
            create_score_distribution_plot(category_results, evaluation_dir, suffix_name, score_key, label, 
                                         evaluator_name, prompt_type, use_argmax=False, eval_target=eval_target, 
                                         correctness_category=category_name)
    
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
        prompt_type (str): Optional prompt type to filter by (e.g., "constitutional_cot", "simple_cot")
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
                # Check if there are any eval_target subdirectories with this evaluator
                for eval_target_dir in prompt_dir.iterdir():
                    if eval_target_dir.is_dir():
                        evaluator_path = eval_target_dir / f"evaluator-{evaluator_name}"
                        if evaluator_path.exists():
                            available_prompt_types.append(prompt_dir.name)
                            break  # Found at least one eval_target, so this prompt_type is valid
        
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
        
        # Find available eval_targets for this prompt type and evaluator
        evaluation_path = Path("evaluation_output") / evaluation_dir / pt
        available_eval_targets = []
        for eval_target_dir in evaluation_path.iterdir():
            if eval_target_dir.is_dir():
                evaluator_path = eval_target_dir / f"evaluator-{evaluator_name}"
                if evaluator_path.exists():
                    available_eval_targets.append(eval_target_dir.name)
        
        if not available_eval_targets:
            print(f"  No eval targets found for evaluator-{evaluator_name} with prompt type {pt}")
            continue
        
        print(f"  Found eval targets: {available_eval_targets}")
        
        # Process each eval target
        for eval_target in available_eval_targets:
            print(f"  Processing eval target: {eval_target}")
            
            # Load results for this evaluator, prompt type, and eval target
            results_by_suffix = load_evaluation_results_by_suffix(evaluation_dir, evaluator_name, pt, eval_target)
            
            if not results_by_suffix:
                print(f"    No data found for evaluator-{evaluator_name} with prompt type {pt} and eval target {eval_target}")
                continue
            
            print(f"    Found {len(results_by_suffix)} suffix conditions: {list(results_by_suffix.keys())}")
            
            # Process each suffix condition
            for suffix_name, results_by_iteration in results_by_suffix.items():
                if suffix_filter and suffix_name != suffix_filter:
                    continue
                    
                print(f"    Processing suffix: {suffix_name}")
                process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name, pt, eval_target)
        
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