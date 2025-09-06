"""
Comprehensive evaluation plotting script for model evaluation structures.

This script handles the model evaluation structure: evaluation_output/evaluation_dir/evaluator-X/iteration-Y/suffix/

It provides all the functionality of both plot_influence.py and the original plot_self_influence.py.
For traditional evaluation, use evaluator "base" which is effectively the same as the original evaluation.
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
    
    evaluator_names = sorted(list(evaluator_names))  # Sort for consistent ordering
    return evaluator_names

def load_evaluation_results_by_suffix(evaluation_dir, evaluator_name="base", prompt_type="cot_prompt"):
    """
    Load model evaluation results organized by suffix condition and prompt type.
    Args:
        evaluation_dir (str): Name of the subfolder in evaluation_output to load from
        evaluator_name (str): Name of the evaluator to load results for (default: "base")
        prompt_type (str): Optional prompt type to filter by (e.g., "training_prompt", "cot_prompt")
    Returns:
        dict: Dictionary mapping suffix conditions to {iteration: results}
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Evaluation directory {evaluation_path} does not exist")
        return {}
    
    # Look for evaluator directory within the specific prompt_type subdirectory
    if prompt_type:
        prompt_type_dir = evaluation_path / prompt_type
        if not prompt_type_dir.exists():
            print(f"Error: Prompt type directory {prompt_type_dir} does not exist")
            return {}
        
        evaluator_path = prompt_type_dir / f"evaluator-{evaluator_name}"
        if not evaluator_path.exists():
            print(f"Error: Evaluator directory {evaluator_path} does not exist")
            return {}
    else:
        # If no prompt_type specified, find the first available one
        evaluator_path = None
        for prompt_type_dir in evaluation_path.iterdir():
            if prompt_type_dir.is_dir():
                potential_evaluator_path = prompt_type_dir / f"evaluator-{evaluator_name}"
                if potential_evaluator_path.exists():
                    evaluator_path = potential_evaluator_path
                    break
        
        if not evaluator_path:
            print(f"Error: Evaluator directory evaluator-{evaluator_name} not found in any prompt_type subdirectory")
            return {}
    
    results_by_suffix = defaultdict(dict)
    
    # Look for suffix directories under the evaluator directory
    # Expected structure: evaluator-X/suffix/iteration-Y/
    suffix_dirs = [d for d in evaluator_path.iterdir() 
                   if d.is_dir() and not d.name.startswith("iteration-")]
    
    if not suffix_dirs:
        print(f"Error: No suffix directories found in {evaluator_path}")
        return {}
    
    print(f"Found {len(suffix_dirs)} suffix directories for evaluator-{evaluator_name}")
    
    for suffix_dir in suffix_dirs:
        suffix_name = suffix_dir.name
        
        # Find all iteration directories under the suffix directory
        iteration_dirs = [d for d in suffix_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("iteration-")]
        
        if not iteration_dirs:
            print(f"Warning: No iteration directories found in suffix {suffix_name}")
            continue
        
        print(f"Found {len(iteration_dirs)} iteration directories for suffix {suffix_name}")
        
        for iteration_dir in iteration_dirs:
            # Extract iteration number from directory name
            iteration_num = int(iteration_dir.name.split("-")[1])
            
            # Look for eval files directly within iteration directories
            # Expected structure: suffix/iteration-X/
            eval_dir = iteration_dir
            
            # Look for eval files (files that end with eval_base.json or similar)
            eval_files = list(eval_dir.glob("*eval*.json"))
            if eval_files:
                # Sort by timestamp (newest first) and take the most recent
                eval_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_eval_file = eval_files[0]
                
                print(f"Loading evaluation results from: {latest_eval_file}")
                try:
                    with open(latest_eval_file, 'r') as f:
                        results = json.load(f)
                    results_by_suffix[suffix_name][iteration_num] = results
                    print(f"  Loaded {len(results)} evaluation examples for {suffix_name} ({prompt_type}) iteration {iteration_num}")
                except Exception as e:
                    print(f"Error loading {latest_eval_file}: {e}")
    
    return results_by_suffix

def analyze_evaluation_results(results_by_iteration, score_key):
    """
    Analyze the loaded model evaluation results for a given score key.
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        score_key (str): Which score to analyze ('full_influence_score', 'reasoning_influence_score', or 'classification_score')
    Returns:
        dict: Summary statistics for each iteration
    """
    summary_stats = {}
    
    for iteration, results in results_by_iteration.items():
        # Handle different result structures
        if isinstance(results, dict) and 'results' in results:
            # New format: results is a dict with 'results' key
            actual_results = results['results']
        else:
            # Old format: results is directly the list
            actual_results = results
        
        # Extract valid scores based on the score_key
        valid_scores = []
        for r in actual_results:
            if score_key == 'classification_score':
                # For classification scores, look in the evaluation field
                score = r.get('evaluation', {}).get('classification_score')
            else:
                # For influence scores, look directly in the result
                score = r.get(score_key)
            
            if score is not None:
                valid_scores.append(score)
        
        if not valid_scores:
            print(f"Warning: No valid scores found for iteration {iteration} and score_key {score_key}")
            continue
        
        # Calculate statistics appropriate for Likert scale data
        q1 = np.percentile(valid_scores, 25)
        q3 = np.percentile(valid_scores, 75)
        iqr = q3 - q1
        
        stats = {
            'total_examples': len(results),
            'valid_scores': len(valid_scores),
            'average_score': np.mean(valid_scores),
            'median_score': np.median(valid_scores),
            'q1_score': q1,
            'q3_score': q3,
            'iqr_score': iqr,
            'std_score': np.std(valid_scores),
            'min_score': np.min(valid_scores),
            'max_score': np.max(valid_scores),
            'score_distribution': {},
            'all_scores': valid_scores
        }
        
        # Score distribution
        for score in range(1, 6):
            count = valid_scores.count(score)
            percentage = (count / len(valid_scores)) * 100
            stats['score_distribution'][score] = {
                'count': count,
                'percentage': percentage
            }
        
        summary_stats[iteration] = stats
        
        print(f"\nIteration {iteration} Model Evaluation Summary for {score_key}:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Valid scores: {stats['valid_scores']}")
        print(f"  Average {score_key} score: {stats['average_score']:.2f}")
        print(f"  Median {score_key} score: {stats['median_score']:.2f}")
        print(f"  IQR: {stats['iqr_score']:.2f} (Q1: {stats['q1_score']:.2f}, Q3: {stats['q3_score']:.2f})")
        print(f"  Score range: {stats['min_score']} - {stats['max_score']}")
        print(f"  Score distribution:")
        for score in range(1, 6):
            dist = stats['score_distribution'][score]
            print(f"    Score {score}: {dist['count']} ({dist['percentage']:.1f}%)")
    
    return summary_stats

def create_eval_plots(summary_stats, evaluation_dir, suffix_name, score_key, label, evaluator_name="base", prompt_type="cot_prompt"):
    """
    Create plots showing evaluation scores across iterations for a given score type and suffix.
    Args:
        summary_stats (dict): Summary statistics for each iteration
        evaluation_dir (str): Name of the evaluation directory for plot titles
        suffix_name (str): Name of the suffix condition
        score_key (str): Which score to plot
        label (str): Label for plot titles and filenames
        evaluator_name (str): Name of the evaluator used
    """
    if not summary_stats:
        print(f"No data to plot for {label} - {suffix_name}!")
        return
    
    # Determine score type directory
    if score_key == "full_influence_score":
        score_type_dir = "whole_response"
    elif score_key == "reasoning_influence_score":
        score_type_dir = "reasoning_only"
    elif score_key == "classification_score":
        score_type_dir = "classification"
    else:
        score_type_dir = "other"
    
    # Create evaluation plots directory with prompt_type structure
    # Structure: plots/experiment_name/prompt_type/evaluator-X/score_type_dir/eval/suffix_name/aggregate
    plots_dir = Path("plots") / evaluation_dir / prompt_type / f"evaluator-{evaluator_name}" / score_type_dir / "eval" / suffix_name / "aggregate"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort iterations for proper ordering
    iterations = sorted(summary_stats.keys())
    
    # Prepare data for plotting
    means = [summary_stats[iter]['average_score'] for iter in iterations]
    medians = [summary_stats[iter]['median_score'] for iter in iterations]
    q1_scores = [summary_stats[iter]['q1_score'] for iter in iterations]
    q3_scores = [summary_stats[iter]['q3_score'] for iter in iterations]
    
    # Create figure with single plot (removed violin plot)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Mean and median plot with shaded interquartile range
    ax.fill_between(iterations, q1_scores, q3_scores, alpha=0.3, color='#2E86AB', label='IQR (Q1-Q3)')
    ax.plot(iterations, means, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Mean')
    ax.plot(iterations, medians, 's-', color='#A23B72', linewidth=2, markersize=6, 
             label='Median', alpha=0.9)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(f'{label} Score', fontsize=12)
    # Create custom title with model name and inference script
    # Always infer model name and inference script from directory structure
    model_name = infer_model_name(evaluation_dir)
    inference_script = infer_inference_script_name(prompt_type)
    title = f'Training method: {model_name}, Inference Prompt: {inference_script}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Add legend
    legend = ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, color='#e9ecef')
    ax.set_ylim(0.5, 5.5)
    ax.set_facecolor('#f8f9fa')
    
    # Add sample size annotations
    for iter in iterations:
        n_samples = len(summary_stats[iter]['all_scores'])
        ax.annotate(f'n={n_samples}', 
                    (iter, 0.7), 
                    ha='center', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "eval_scores.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved evaluation plot to: {plot_path}")
    plt.close()

def get_weighted_evaluation_score(entry, result_key="full_influence_result"):
    """
    Parse the probabilities from entry[result_key] and return the weighted average Likert score.
    """
    s = entry.get(result_key, '')
    if 'probs:' in s:
        try:
            probs_str = s.split('probs:')[1].strip()
            if probs_str.endswith(')'):
                probs_str = probs_str[:-1]
            probs = ast.literal_eval(probs_str)
            weighted = sum(int(k) * float(v) for k, v in probs.items())
            return weighted
        except Exception as e:
            print(f'Error parsing {result_key}: {s} ({e})')
    return None

def plot_evaluation_by_example(results_by_iteration, evaluation_dir, suffix_name, score_key, label, use_weighted=False, result_key="full_influence_result", evaluator_name="base", prompt_type="cot_prompt"):
    """
    For each example_index, plot the model evaluation score for each iteration.
    Each example_index gets its own line, showing how its score changes over time.
    """
    example_scores = defaultdict(lambda: defaultdict(list))
    iterations = sorted(results_by_iteration.keys())
    
    for iteration in iterations:
        results = results_by_iteration[iteration]
        
        # Handle different result structures
        if isinstance(results, dict) and 'results' in results:
            # New format: results is a dict with 'results' key
            actual_results = results['results']
        else:
            # Old format: results is directly the list
            actual_results = results
        
        for r in actual_results:
            idx = r.get('example_index', r.get('idx', None))
            if use_weighted:
                score = get_weighted_evaluation_score(r, result_key=result_key)
            else:
                # Extract score based on score_key type
                if score_key == 'classification_score':
                    score = r.get('evaluation', {}).get('classification_score')
                else:
                    score = r.get(score_key, None)
            
            if idx is not None and score is not None:
                example_scores[idx][iteration].append(score)
    
    # Calculate means for each example across iterations
    example_means = {}
    for idx, iter_dict in example_scores.items():
        example_means[idx] = [np.mean(iter_dict[iteration]) if iteration in iter_dict and len(iter_dict[iteration]) > 0 else np.nan 
                             for iteration in iterations]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    n_classes = len(example_means)
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    elif n_classes <= 20:
        palette = sns.color_palette("tab20", n_classes)
    else:
        palette = sns.color_palette("hls", n_classes)
    
    # Plot lines for all examples
    for i, (idx, means) in enumerate(sorted(example_means.items())):
        if use_weighted:
            # For weighted plots, don't add labels (will use colorbar instead)
            ax.plot(iterations, means, color=palette[i % len(palette)], alpha=0.7)
        else:
            # For non-weighted plots, add labels for individual legend
            ax.plot(iterations, means, label=f'{idx}', color=palette[i % len(palette)], alpha=0.7)
    
    ax.set_facecolor('#ffffff')
    plt.xlabel('Iteration', fontsize=12)
    
    if use_weighted:
        ylabel = f'Weighted Mean {label} Score'
        # Create custom title with model name and inference script
        # Always infer model name and inference script from directory structure
        model_name = infer_model_name(evaluation_dir)
        inference_script = infer_inference_script_name(prompt_type)
        title = f'Training method: {model_name}\nInference Prompt: {inference_script}'
        
        # Create color spectrum legend instead of individual entries
        if example_means:
            min_idx = min(example_means.keys())
            max_idx = max(example_means.keys())
            
            # Create a colorbar-like legend
            from matplotlib.patches import Rectangle
            from matplotlib.colors import Normalize
            
            # Create a small colorbar on the right side
            norm = Normalize(min_idx, max_idx)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            
            # Add colorbar with custom positioning
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(f'Example Index\n({min_idx} to {max_idx})', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
            
    else:
        ylabel = f'Mean {label} Score'
        # Create custom title with model name and inference script
        # Always infer model name and inference script from directory structure
        model_name = infer_model_name(evaluation_dir)
        inference_script = infer_inference_script_name(prompt_type)
        title = f'Training method: {model_name}\nInference Prompt: {inference_script}'
        # Add individual legend for non-weighted plots
        plt.legend(title='Example Index', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, axis='both', alpha=0.5, color='#bbbbbb')
    plt.ylim(0.5, 5.5)
    plt.xticks(iterations)
    plt.tight_layout()
    
    # Save the plot
    if score_key == "full_influence_score":
        score_type_dir = "whole_response"
    elif score_key == "reasoning_influence_score":
        score_type_dir = "reasoning_only"
    else:
        score_type_dir = "other"
    
    base_dir = Path("plots") / evaluation_dir / prompt_type / f"evaluator-{evaluator_name}" / score_type_dir / "eval" / suffix_name / "by_example"
    
    if use_weighted:
        sub_dir = base_dir / "weighted"
        plot_filename = "by_example_weighted.png"
    else:
        sub_dir = base_dir / "argmax"
        plot_filename = "by_example_argmax.png"
    
    sub_dir.mkdir(parents=True, exist_ok=True)
    plot_path = sub_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    print(f"\nSaved model evaluation by example plot to: {plot_path}")
    plt.close()

def create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name="base", prompt_type="cot_prompt", use_argmax=True):
    """
    Create a stacked bar chart showing the distribution of scores across iterations.
    """
    iterations = sorted(results_by_iteration.keys())
    
    # Count scores for each iteration
    score_counts = {}
    for iteration in iterations:
        score_counts[iteration] = {-1: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        results = results_by_iteration[iteration]
        
        # Handle different result structures
        if isinstance(results, dict) and 'results' in results:
            # New format: results is a dict with 'results' key
            actual_results = results['results']
        else:
            # Old format: results is directly the list
            actual_results = results
        
        print(f"    Processing iteration {iteration}: {len(actual_results)} results, score_key: {score_key}")
        
        for r in actual_results:
            score_to_use = None
            
            # Try to extract from probability distribution first (for both argmax and weighted average)
            result_key = score_key.replace('_score', '_result')
            result_str = r.get(result_key, '')
            
            if isinstance(result_str, str) and 'probs:' in result_str:
                try:
                    # Extract probability dict from string like "Score: 5 (probs: {'1': 0.0, '2': 0.1, ...})"
                    import re
                    probs_match = re.search(r"probs:\s*(\{[^}]+\})", result_str)
                    if probs_match:
                        probs_str = probs_match.group(1)
                        probs_dict = eval(probs_str)  # Safe here since we control the format
                        
                        if use_argmax:
                            # For argmax version: find the score with highest probability
                            argmax_score = max(probs_dict.items(), key=lambda x: float(x[1]))[0]
                            score_to_use = int(argmax_score)
                        else:
                            # For weighted average version: calculate weighted sum
                            weighted_sum = sum(int(score_val) * float(prob) for score_val, prob in probs_dict.items())
                            score_to_use = round(weighted_sum)
                except Exception as e:
                    pass  # Fall back to clean integer score
            
            # If we couldn't extract from probability distribution, use the clean integer score
            if score_to_use is None:
                if score_key == 'classification_score':
                    score_to_use = r.get('evaluation', {}).get('classification_score')
                else:
                    score_to_use = r.get(score_key, None)
            
            if score_to_use is not None and score_to_use in [-1, 1, 2, 3, 4, 5]:
                score_counts[iteration][score_to_use] += 1
            else:
                print(f"      Skipping result with score: {score_to_use} (type: {type(score_to_use)})")
        
        print(f"    Iteration {iteration} score counts: {score_counts[iteration]}")
    
    # Prepare data for stacked bar chart
    scores_neg1 = [score_counts[iter][-1] for iter in iterations]
    scores_1 = [score_counts[iter][1] for iter in iterations]
    scores_2 = [score_counts[iter][2] for iter in iterations]
    scores_3 = [score_counts[iter][3] for iter in iterations]
    scores_4 = [score_counts[iter][4] for iter in iterations]
    scores_5 = [score_counts[iter][5] for iter in iterations]
    
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
    
    # Create bars in stacking order (Score 5 at top, No Score at bottom) with matching legend order
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
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Examples', fontsize=12)
    # Create custom title with model name and inference script
    # Always infer model name and inference script from directory structure
    model_name = infer_model_name(evaluation_dir)
    inference_script = infer_inference_script_name(prompt_type)
    title = f'Training method: {model_name}, Inference Prompt: {inference_script}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis ticks to show every iteration
    ax.set_xticks(iterations)
    ax.set_xticklabels(iterations)
    
    # Create legend with explicit order (Score 5 at top, No Score at bottom)
    handles = [p5, p4, p3, p2, p1, p_neg1]
    labels = ['Score 5', 'Score 4', 'Score 3', 'Score 2', 'Score 1', 'No Score']
    
    legend = ax.legend(
        handles, labels,
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
    else:
        score_type_dir = "other"
    
    method_subdir = "argmax" if use_argmax else "weighted_avg"
    plots_dir = Path("plots") / evaluation_dir / prompt_type / f"evaluator-{evaluator_name}" / score_type_dir / "eval" / suffix_name / "distribution" / method_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "score_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved score distribution plot to: {plot_path}")
    plt.close()

def create_weighted_score_comparison_plot(weighted_scores_by_suffix, evaluation_dir, label, filename, evaluator_name, prompt_type):
    """
    Create a weighted score comparison plot across suffixes for a specific evaluator.
    Args:
        weighted_scores_by_suffix (dict): Dictionary mapping suffix to {iteration: weighted_score}
        evaluation_dir (str): Name of the evaluation directory
        label (str): Label for the plot
        filename (str): Filename to save the plot
        evaluator_name (str): Name of the evaluator
    """
    plt.figure(figsize=(10, 6))
    
    for suffix, scores_by_iter in weighted_scores_by_suffix.items():
        iterations = sorted(scores_by_iter.keys())
        means = [scores_by_iter[it] for it in iterations]
        plt.plot(iterations, means, marker='o', label=suffix.replace('_', ' ').title())
    
    plt.xlabel('Iteration')
    plt.ylabel(f'Average Weighted Influence Score ({label})')
    plt.title(f'Weighted Influence Score Comparison Across Suffixes\n{label} - {evaluation_dir} - evaluator-{evaluator_name}')
    plt.legend(title='Suffix Condition')
    plt.ylim(0.5, 5.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_dir = Path('plots') / evaluation_dir / prompt_type / f'evaluator-{evaluator_name}' / 'weighted_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300)
    print(f"\nSaved weighted comparison plot to: {out_dir / filename}")
    plt.close()

def create_cross_evaluator_comparison(evaluation_dir, evaluators, suffix_name, score_key, label, prompt_type="cot_prompt"):
    """
    Create a comparison plot showing all evaluators on the same plot.
    Args:
        evaluation_dir (str): Name of the evaluation directory
        evaluators (list): List of evaluator names to compare
        suffix_name (str): Name of the suffix condition
        score_key (str): Which score to plot
        label (str): Label for plot titles
        prompt_type (str): Prompt type to analyze (e.g., "training_prompt", "cot_prompt")
    """
    if len(evaluators) < 2:
        print(f"Need at least 2 evaluators for comparison, got {len(evaluators)}")
        return
    
    # Load data for all evaluators
    evaluator_data = {}
    for evaluator_name in evaluators:
        results_by_suffix = load_evaluation_results_by_suffix(evaluation_dir, evaluator_name, prompt_type)
        if suffix_name in results_by_suffix:
            summary_stats = analyze_evaluation_results(results_by_suffix[suffix_name], score_key)
            if summary_stats:
                evaluator_data[evaluator_name] = summary_stats
    
    if not evaluator_data:
        print(f"No data found for suffix {suffix_name} across evaluators")
        return
    
    # Determine score type directory
    if score_key == "full_influence_score":
        score_type_dir = "whole_response"
    elif score_key == "reasoning_influence_score":
        score_type_dir = "reasoning_only"
    elif score_key == "classification_score":
        score_type_dir = "classification"
    else:
        score_type_dir = "other"
    
    # Create comparison plots directory with prompt_type structure
    plots_dir = Path("plots") / evaluation_dir / prompt_type / "cross_evaluator_comparison" / score_type_dir / suffix_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Color palette for different evaluators
    colors = sns.color_palette("Set1", len(evaluator_data))
    
    for i, (evaluator_name, summary_stats) in enumerate(evaluator_data.items()):
        # Sort iterations for proper ordering
        iterations = sorted(summary_stats.keys())
        
        # Prepare data for plotting
        means = [summary_stats[iter]['average_score'] for iter in iterations]
        medians = [summary_stats[iter]['median_score'] for iter in iterations]
        
        # Plot lines for this evaluator
        ax.plot(iterations, means, 'o-', color=colors[i], linewidth=2, markersize=8, 
                label=f'evaluator-{evaluator_name} (mean)', alpha=0.8)
        ax.plot(iterations, medians, 's--', color=colors[i], linewidth=2, markersize=6, 
                label=f'evaluator-{evaluator_name} (median)', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(f'{label} Score', fontsize=12)
    ax.set_title(f'Cross-Evaluator Comparison: {label} Scores\n{suffix_name} - {evaluation_dir}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, color='#e9ecef')
    ax.set_ylim(0.5, 5.5)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"cross_evaluator_comparison_{score_key}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved cross-evaluator comparison plot to: {plot_path}")
    plt.close()





def process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name, prompt_type="cot_prompt"):
    """
    Process a single suffix condition and create all its plots.
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
    # For this evaluation, we have evaluation scores
    score_keys_and_labels = [
        ("full_evaluation_score", "Full Response Evaluation"),
        ("reasoning_evaluation_score", "Reasoning Only Evaluation")
    ]
    
    # Process each score type
    for score_key, label in score_keys_and_labels:
        print(f"    Analyzing {label} results...")
        
        # Analyze results
        summary_stats = analyze_evaluation_results(results_by_iteration, score_key)
        
        if not summary_stats:
            print(f"      No summary stats for {label}")
            continue
        
        print(f"      Found {len(summary_stats)} iterations for {label}")
        
        # Create all plots for this score type
        print(f"      Creating aggregate plot for {label}...")
        create_eval_plots(summary_stats, evaluation_dir, suffix_name, score_key, label, evaluator_name, prompt_type)
        
        print(f"      Creating distribution plots for {label}...")
        # Create argmax version
        print(f"        Creating argmax distribution plot...")
        create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name, prompt_type, True)
        # Create weighted average version  
        print(f"        Creating weighted average distribution plot...")
        create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name, prompt_type, False)
        
        print(f"      Creating by-example plot for {label}...")
        plot_evaluation_by_example(results_by_iteration, evaluation_dir, suffix_name, score_key, label, use_weighted=False, evaluator_name=evaluator_name, prompt_type=prompt_type)
    
    plt.close('all')  # Close all plots to free memory
    
    print(f"    Completed all plots for {suffix_name}")
    return f"Completed all plots for {suffix_name}"

def process_evaluator(evaluation_dir, evaluator_name, suffix_filter=None, prompt_type=None):
    """
    Process results for a single evaluator.
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
    if prompt_type is None:
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
        
        print(f"Found prompt types for evaluator-{evaluator_name}: {available_prompt_types}")
        prompt_types_to_process = available_prompt_types
    else:
        prompt_types_to_process = [prompt_type]
    
    # Process each prompt type
    for pt in prompt_types_to_process:
        print(f"\nProcessing prompt type: {pt}")
        
        # Load model evaluation results organized by suffix and prompt type
        results_by_suffix = load_evaluation_results_by_suffix(evaluation_dir, evaluator_name, pt)
    
        if not results_by_suffix:
            print(f"No model evaluation results found for evaluator-{evaluator_name} with prompt type {pt}!")
            continue
        
        # Filter to specific suffix if requested
        if suffix_filter:
            if suffix_filter not in results_by_suffix:
                print(f"Suffix '{suffix_filter}' not found in evaluator-{evaluator_name} with prompt type {pt}. Available suffixes: {list(results_by_suffix.keys())}")
                continue
            results_by_suffix = {suffix_filter: results_by_suffix[suffix_filter]}
        
        print(f"Found data for {len(results_by_suffix)} suffix conditions: {list(results_by_suffix.keys())}")
        
        # Process suffix conditions sequentially
        print(f"Processing {len(results_by_suffix)} suffix conditions...")
        print(f"This will create individual plots for each suffix condition...")
        
        for suffix_name, results_by_iteration in results_by_suffix.items():
            print(f"\n{'-'*60}")
            print(f"PROCESSING SUFFIX CONDITION: {suffix_name} (evaluator-{evaluator_name}, prompt type: {pt})")
            print(f"  Results by iteration: {list(results_by_iteration.keys())}")
            print(f"  Total examples across all iterations: {sum(len(results) for results in results_by_iteration.values())}")
            print(f"{'-'*60}")
            
            process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name, pt)
        
        # Create weighted score comparison plots across suffixes for this evaluator and prompt type
        if len(results_by_suffix) > 1:
            print(f"\nCreating weighted score comparison plots across suffixes for evaluator-{evaluator_name} (prompt type: {pt})...")
            
            weighted_scores_by_suffix_full = {}
            weighted_scores_by_suffix_reasoning = {}
            
            for suffix_name, results_by_iteration in results_by_suffix.items():
                weighted_means_full = {}
                weighted_means_reasoning = {}
                
                for iteration, results in results_by_iteration.items():
                    weighted_scores_full = []
                    weighted_scores_reasoning = []
                    
                    for r in results:
                        w_full = get_weighted_evaluation_score(r, result_key="full_influence_result")
                        w_reasoning = get_weighted_evaluation_score(r, result_key="reasoning_influence_result")
                        if w_full is not None:
                            weighted_scores_full.append(w_full)
                        if w_reasoning is not None:
                            weighted_scores_reasoning.append(w_reasoning)
                    
                    if weighted_scores_full:
                        weighted_means_full[iteration] = np.mean(weighted_scores_full)
                    if weighted_scores_reasoning:
                        weighted_means_reasoning[iteration] = np.mean(weighted_scores_reasoning)
                
                weighted_scores_by_suffix_full[suffix_name] = weighted_means_full
                weighted_scores_by_suffix_reasoning[suffix_name] = weighted_means_reasoning
            
            # Create weighted comparison plots
            if weighted_scores_by_suffix_full:
                create_weighted_score_comparison_plot(weighted_scores_by_suffix_full, evaluation_dir, 
                                                     "Full Response", "weighted_score_comparison_full.png", evaluator_name, pt)
        
        print(f"\nCompleted processing evaluator-{evaluator_name} with prompt type {pt}!")
    
    print(f"\nCompleted processing evaluator-{evaluator_name}!")

def evaluator_worker(args):
    """
    Worker function to process a single evaluator in parallel.
    Args:
        args: Tuple containing (evaluation_dir, evaluator_name, suffix_filter, prompt_type)
    """
    evaluation_dir, evaluator_name, suffix_filter, prompt_type = args
    
    try:
        process_evaluator(evaluation_dir, evaluator_name, suffix_filter, prompt_type)
        return f"Completed evaluator-{evaluator_name}"
    except Exception as e:
        return f"Error processing evaluator-{evaluator_name}: {str(e)}"



def main():
    parser = argparse.ArgumentParser(description='Plot model evaluation results by suffix condition (use evaluator "base" for traditional evaluation)')
    parser.add_argument('evaluation_dir', type=str, help='Evaluation directory name')
    parser.add_argument('--suffix', type=str, help='Specific suffix condition to analyze (optional)')
    parser.add_argument('--evaluator', type=str, help='Specific evaluator name (default: process all evaluators)')
    parser.add_argument('--prompt-type', type=str, help='Prompt type to analyze (e.g., "training_prompt", "cot_prompt")')
    parser.add_argument('--list-evaluators', action='store_true', help='List available evaluators and exit')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing (use sequential processing)')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker processes (default: auto)')
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
    
    # Determine if multiprocessing should be used
    use_multiprocessing = not args.no_multiprocessing and len(evaluators_to_process) > 1
    
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
        
        if use_multiprocessing:
            print(f"Using multiprocessing to process {len(evaluators_to_process)} evaluators in parallel...")
            
            # Prepare tasks for multiprocessing
            evaluator_tasks = [
                (evaluation_dir, evaluator_name, args.suffix, pt)
                for evaluator_name in evaluators_to_process
            ]
            
            # Determine number of workers
            max_workers = args.max_workers or min(mp.cpu_count(), len(evaluator_tasks))
            print(f"Using {max_workers} worker processes")
            
            # Set multiprocessing start method for compatibility
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
            
            # Process evaluators in parallel
            with mp.Pool(max_workers) as pool:
                results = pool.map(evaluator_worker, evaluator_tasks)
            
            for result in results:
                print(f"  {result}")
        else:
            # Process evaluators sequentially
            print(f"Processing evaluators sequentially...")
            for i, evaluator_name in enumerate(evaluators_to_process):
                print(f"\n[{i+1}/{len(evaluators_to_process)}] About to process evaluator: {evaluator_name}")
                process_evaluator(evaluation_dir, evaluator_name, args.suffix, pt)
                print(f"[{i+1}/{len(evaluators_to_process)}] Completed processing evaluator: {evaluator_name}")
        
        # Create cross-evaluator comparison plots if we have multiple evaluators
        if len(evaluators_to_process) > 1:
            print(f"\n{'='*80}")
            print(f"CREATING CROSS-EVALUATOR COMPARISONS FOR PROMPT TYPE: {pt}")
            print(f"{'='*80}")
            
            # Find all suffix conditions that exist across evaluators
            all_suffixes = set()
            for evaluator_name in evaluators_to_process:
                results_by_suffix = load_evaluation_results_by_suffix(evaluation_dir, evaluator_name, pt)
                all_suffixes.update(results_by_suffix.keys())
            
            # Filter to specific suffix if requested
            if args.suffix:
                if args.suffix in all_suffixes:
                    all_suffixes = {args.suffix}
                else:
                    print(f"Suffix '{args.suffix}' not found in any evaluator")
                    all_suffixes = set()
            
            # Create comparison plots for each suffix and score type
            for suffix_name in sorted(all_suffixes):
                print(f"\nCreating cross-evaluator comparison for suffix: {suffix_name}")
                
                for score_key, label in [
                    ("full_influence_score", "Full Response"),
                    ("reasoning_influence_score", "Reasoning Only")
                ]:
                    print(f"  Creating comparison for {label} scores...")
                    create_cross_evaluator_comparison(
                        evaluation_dir, evaluators_to_process, suffix_name, score_key, label, pt
                    )
    
    # Print completion message
    print(f"\n{'='*80}")
    print(f"EVALUATION PLOTTING COMPLETE!")
    print(f"{'='*80}")
    print(f"Evaluation directory: {evaluation_dir}")
    print(f"Processed evaluators: {evaluators_to_process}")
    print(f"Processed prompt types: {prompt_types_to_process}")
    if args.suffix:
        print(f"Filtered to suffix: {args.suffix}")
    print(f"Individual plots saved to: plots/{evaluation_dir}/*/evaluator-*/")
    if len(evaluators_to_process) > 1:
        print(f"Cross-evaluator comparisons saved to: plots/{evaluation_dir}/*/cross_evaluator_comparison/")

if __name__ == "__main__":
    main() 