"""
Comprehensive evaluation plotting script for self-evaluation structures.

This script handles the self-evaluation structure: evaluation_output/evaluation_dir/evaluator-X/iteration-Y/suffix/

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
    
    evaluator_dirs = [d for d in evaluation_path.iterdir() 
                     if d.is_dir() and d.name.startswith("evaluator-")]
    
    evaluator_names = [d.name.replace("evaluator-", "") for d in evaluator_dirs]
    evaluator_names.sort()  # Sort for consistent ordering
    
    return evaluator_names

def load_self_evaluation_results_by_suffix(evaluation_dir, evaluator_name="base"):
    """
    Load self-evaluation results organized by suffix condition.
    Args:
        evaluation_dir (str): Name of the subfolder in evaluation_output to load from
        evaluator_name (str): Name of the evaluator to load results for (default: "base")
    Returns:
        dict: Dictionary mapping suffix conditions to {iteration: results}
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Evaluation directory {evaluation_path} does not exist")
        return {}
    
    # Look for evaluator directory
    evaluator_path = evaluation_path / f"evaluator-{evaluator_name}"
    if not evaluator_path.exists():
        print(f"Error: Evaluator directory {evaluator_path} does not exist")
        return {}
    
    results_by_suffix = defaultdict(dict)
    
    # Find all iteration directories under the evaluator directory
    iteration_dirs = [d for d in evaluator_path.iterdir() 
                     if d.is_dir() and d.name.startswith("iteration-")]
    
    if not iteration_dirs:
        print(f"Error: No iteration directories found in {evaluator_path}")
        return {}
    
    print(f"Found {len(iteration_dirs)} iteration directories for evaluator-{evaluator_name}")
    
    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        iteration_num = int(iteration_dir.name.split("-")[1])
        
        # Look for suffix directories within iteration directories
        for suffix_dir in iteration_dir.iterdir():
            if suffix_dir.is_dir():
                suffix_name = suffix_dir.name
                # Look for eval files directly in suffix directory (no self_eval subdirectory)
                eval_files = list(suffix_dir.glob("eval_*.json"))
                if eval_files:
                    # Sort by timestamp (newest first) and take the most recent
                    eval_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    latest_eval_file = eval_files[0]
                    
                    print(f"Loading evaluation results from: {latest_eval_file}")
                    try:
                        with open(latest_eval_file, 'r') as f:
                            results = json.load(f)
                        results_by_suffix[suffix_name][iteration_num] = results
                        print(f"  Loaded {len(results)} evaluation examples for {suffix_name} iteration {iteration_num}")
                    except Exception as e:
                        print(f"Error loading {latest_eval_file}: {e}")
                        continue
    
    return results_by_suffix

def analyze_self_eval_results(results_by_iteration, score_key):
    """
    Analyze the loaded self-evaluation results for a given score key.
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        score_key (str): Which score to analyze ('full_influence_score' or 'reasoning_influence_score')
    Returns:
        dict: Summary statistics for each iteration
    """
    summary_stats = {}
    
    for iteration, results in results_by_iteration.items():
        # Extract valid influence scores
        valid_scores = [r.get(score_key, None) for r in results if r.get(score_key, None) is not None]
        
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
        
        print(f"\nIteration {iteration} Self-Evaluation Summary for {score_key}:")
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

def create_eval_plots(summary_stats, evaluation_dir, suffix_name, score_key, label, evaluator_name="base"):
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
    else:
        score_type_dir = "other"
    
    # Create evaluation plots directory
    plots_dir = Path("plots") / evaluation_dir / f"evaluator-{evaluator_name}" / score_type_dir / "eval" / suffix_name / "aggregate"
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
    ax.set_title(f'Evaluation: {label} Scores Across Iterations\n{suffix_name} - {evaluation_dir} - evaluator-{evaluator_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, color='#e9ecef')
    ax.set_ylim(0.5, 5.5)
    ax.set_facecolor('#f8f9fa')
    
    # Add value labels on points
    for i, (iter, mean, q1, q3) in enumerate(zip(iterations, means, q1_scores, q3_scores)):
        ax.annotate(f'{mean:.2f} (IQR: {q3-q1:.2f})', 
                    (iter, q3 + 0.1), 
                    ha='center', va='bottom', fontsize=9)
    
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

def get_weighted_self_eval_score(entry, result_key="full_influence_result"):
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

def plot_self_eval_by_example(results_by_iteration, evaluation_dir, suffix_name, score_key, label, use_weighted=False, result_key="full_influence_result", evaluator_name="base"):
    """
    For each example_index, plot the self-evaluation score for each iteration.
    Each example_index gets its own line, showing how its score changes over time.
    """
    example_scores = defaultdict(lambda: defaultdict(list))
    iterations = sorted(results_by_iteration.keys())
    
    for iteration in iterations:
        results = results_by_iteration[iteration]
        for r in results:
            idx = r.get('example_index', r.get('idx', None))
            if use_weighted:
                score = get_weighted_self_eval_score(r, result_key=result_key)
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
    
    for i, (idx, means) in enumerate(sorted(example_means.items())):
        ax.plot(iterations, means, label=f'{idx}', color=palette[i % len(palette)], alpha=0.7)
    
    ax.set_facecolor('#ffffff')
    plt.xlabel('Iteration', fontsize=12)
    
    if use_weighted:
        ylabel = f'Weighted Mean {label} Score'
        title = f'Self-Evaluation: Weighted Mean {label} Score by Example\n{suffix_name} - {evaluation_dir} - evaluator-{evaluator_name}'
    else:
        ylabel = f'Mean {label} Score'
        title = f'Self-Evaluation: Mean {label} Score by Example\n{suffix_name} - {evaluation_dir} - evaluator-{evaluator_name}'
    
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title='Example Index', bbox_to_anchor=(1.05, 1), loc='upper left')
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
    
    base_dir = Path("plots") / evaluation_dir / f"evaluator-{evaluator_name}" / score_type_dir / "self_eval" / suffix_name / "by_example"
    
    if use_weighted:
        sub_dir = base_dir / "weighted"
        plot_filename = "by_example_weighted.png"
    else:
        sub_dir = base_dir / "argmax"
        plot_filename = "by_example_argmax.png"
    
    sub_dir.mkdir(parents=True, exist_ok=True)
    plot_path = sub_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    print(f"\nSaved self-evaluation by example plot to: {plot_path}")
    plt.close()

def create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name="base"):
    """
    Create a stacked bar chart showing the distribution of scores across iterations.
    """
    iterations = sorted(results_by_iteration.keys())
    
    # Count scores for each iteration
    score_counts = {}
    for iteration in iterations:
        score_counts[iteration] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        results = results_by_iteration[iteration]
        for r in results:
            score = r.get(score_key, None)
            if score is not None and score in range(1, 6):
                score_counts[iteration][score] += 1
    
    # Prepare data for stacked bar chart
    scores_1 = [score_counts[iter][1] for iter in iterations]
    scores_2 = [score_counts[iter][2] for iter in iterations]
    scores_3 = [score_counts[iter][3] for iter in iterations]
    scores_4 = [score_counts[iter][4] for iter in iterations]
    scores_5 = [score_counts[iter][5] for iter in iterations]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.6
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4']
    
    p1 = ax.bar(iterations, scores_1, width, label='Score 1', color=colors[0])
    p2 = ax.bar(iterations, scores_2, width, bottom=scores_1, label='Score 2', color=colors[1])
    p3 = ax.bar(iterations, scores_3, width, bottom=np.array(scores_1) + np.array(scores_2), 
                label='Score 3', color=colors[2])
    p4 = ax.bar(iterations, scores_4, width, 
                bottom=np.array(scores_1) + np.array(scores_2) + np.array(scores_3), 
                label='Score 4', color=colors[3])
    p5 = ax.bar(iterations, scores_5, width, 
                bottom=np.array(scores_1) + np.array(scores_2) + np.array(scores_3) + np.array(scores_4), 
                label='Score 5', color=colors[4])
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Examples', fontsize=12)
    ax.set_title(f'Self-Evaluation: {label} Score Distribution Across Iterations\n{suffix_name} - {evaluation_dir} - evaluator-{evaluator_name}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Save the plot
    if score_key == "full_influence_score":
        score_type_dir = "whole_response"
    elif score_key == "reasoning_influence_score":
        score_type_dir = "reasoning_only"
    else:
        score_type_dir = "other"
    
    plots_dir = Path("plots") / evaluation_dir / f"evaluator-{evaluator_name}" / score_type_dir / "self_eval" / suffix_name / "distribution"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "score_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved score distribution plot to: {plot_path}")
    plt.close()

def create_weighted_score_comparison_plot(weighted_scores_by_suffix, evaluation_dir, label, filename, evaluator_name):
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
    
    out_dir = Path('plots') / evaluation_dir / f'evaluator-{evaluator_name}' / 'weighted_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / filename, dpi=300)
    print(f"\nSaved weighted comparison plot to: {out_dir / filename}")
    plt.close()

def create_cross_evaluator_comparison(evaluation_dir, evaluators, suffix_name, score_key, label):
    """
    Create a comparison plot showing all evaluators on the same plot.
    Args:
        evaluation_dir (str): Name of the evaluation directory
        evaluators (list): List of evaluator names to compare
        suffix_name (str): Name of the suffix condition
        score_key (str): Which score to plot
        label (str): Label for plot titles
    """
    if len(evaluators) < 2:
        print(f"Need at least 2 evaluators for comparison, got {len(evaluators)}")
        return
    
    # Load data for all evaluators
    evaluator_data = {}
    for evaluator_name in evaluators:
        results_by_suffix = load_self_evaluation_results_by_suffix(evaluation_dir, evaluator_name)
        if suffix_name in results_by_suffix:
            summary_stats = analyze_self_eval_results(results_by_suffix[suffix_name], score_key)
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
    else:
        score_type_dir = "other"
    
    # Create comparison plots directory
    plots_dir = Path("plots") / evaluation_dir / "cross_evaluator_comparison" / score_type_dir / suffix_name
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





def process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name):
    """
    Process a single suffix condition and create all its plots.
    Args:
        suffix_name (str): Name of the suffix condition
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator
    """
    print(f"  Processing suffix: {suffix_name}")
    
    if not results_by_iteration:
        print(f"    No data found for suffix: {suffix_name}")
        return
    
    # Process both score types
    for score_key, label, result_key in [
        ("full_influence_score", "Full Response", "full_influence_result"),
        ("reasoning_influence_score", "Reasoning Only", "reasoning_influence_result")
    ]:
        print(f"    Analyzing {label} results...")
        
        # Analyze results
        summary_stats = analyze_self_eval_results(results_by_iteration, score_key)
        
        if not summary_stats:
            print(f"    No summary stats for {label}")
            continue
        
        print(f"    Found {len(summary_stats)} iterations for {label}")
        
        # Create all plots for this score type
        print(f"    Creating aggregate plot for {label}...")
        create_eval_plots(summary_stats, evaluation_dir, suffix_name, score_key, label, evaluator_name)
        
        print(f"    Creating distribution plot for {label}...")
        create_score_distribution_plot(results_by_iteration, evaluation_dir, suffix_name, score_key, label, evaluator_name)
        
        print(f"    Creating by-example argmax plot for {label}...")
        plot_self_eval_by_example(results_by_iteration, evaluation_dir, suffix_name, score_key, label, use_weighted=False, evaluator_name=evaluator_name)
        
        print(f"    Creating by-example weighted plot for {label}...")
        plot_self_eval_by_example(results_by_iteration, evaluation_dir, suffix_name, score_key, label, use_weighted=True, result_key=result_key, evaluator_name=evaluator_name)
        
        plt.close('all')  # Close all plots to free memory
    
    print(f"    Completed all plots for {suffix_name}")
    return f"Completed all plots for {suffix_name}"

def process_evaluator(evaluation_dir, evaluator_name, suffix_filter=None):
    """
    Process results for a single evaluator.
    Args:
        evaluation_dir (str): Name of the evaluation directory
        evaluator_name (str): Name of the evaluator to process
        suffix_filter (str): Optional suffix to filter to
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING EVALUATOR: {evaluator_name}")
    print(f"{'='*80}")
    
    # Load self-evaluation results organized by suffix
    results_by_suffix = load_self_evaluation_results_by_suffix(evaluation_dir, evaluator_name)
    
    if not results_by_suffix:
        print(f"No self-evaluation results found for evaluator-{evaluator_name}!")
        return
    
    # Filter to specific suffix if requested
    if suffix_filter:
        if suffix_filter not in results_by_suffix:
            print(f"Suffix '{suffix_filter}' not found in evaluator-{evaluator_name}. Available suffixes: {list(results_by_suffix.keys())}")
            return
        results_by_suffix = {suffix_filter: results_by_suffix[suffix_filter]}
    
    print(f"Found data for {len(results_by_suffix)} suffix conditions: {list(results_by_suffix.keys())}")
    
    # Process suffix conditions sequentially
    print(f"Processing {len(results_by_suffix)} suffix conditions...")
    print(f"This will create individual plots for each suffix condition...")
    
    for suffix_name, results_by_iteration in results_by_suffix.items():
        print(f"\n{'-'*60}")
        print(f"PROCESSING SUFFIX CONDITION: {suffix_name} (evaluator-{evaluator_name})")
        print(f"  Results by iteration: {list(results_by_iteration.keys())}")
        print(f"  Total examples across all iterations: {sum(len(results) for results in results_by_iteration.values())}")
        print(f"{'-'*60}")
        
        process_suffix_condition(suffix_name, results_by_iteration, evaluation_dir, evaluator_name)
    
    # Create weighted score comparison plots across suffixes for this evaluator
    if len(results_by_suffix) > 1:
        print(f"\nCreating weighted score comparison plots across suffixes for evaluator-{evaluator_name}...")
        
        weighted_scores_by_suffix_full = {}
        weighted_scores_by_suffix_reasoning = {}
        
        for suffix_name, results_by_iteration in results_by_suffix.items():
            weighted_means_full = {}
            weighted_means_reasoning = {}
            
            for iteration, results in results_by_iteration.items():
                weighted_scores_full = []
                weighted_scores_reasoning = []
                
                for r in results:
                    w_full = get_weighted_self_eval_score(r, result_key="full_influence_result")
                    w_reasoning = get_weighted_self_eval_score(r, result_key="reasoning_influence_result")
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
                                                 "Full Response", "weighted_score_comparison_full.png", evaluator_name)
        
        if weighted_scores_by_suffix_reasoning:
            create_weighted_score_comparison_plot(weighted_scores_by_suffix_reasoning, evaluation_dir, 
                                                 "Reasoning Only", "weighted_score_comparison_reasoning.png", evaluator_name)
    
    print(f"\nCompleted processing evaluator-{evaluator_name}!")
    print(f"Processed suffix conditions: {list(results_by_suffix.keys())}")

def evaluator_worker(args):
    """
    Worker function to process a single evaluator in parallel.
    Args:
        args: Tuple containing (evaluation_dir, evaluator_name, suffix_filter)
    """
    evaluation_dir, evaluator_name, suffix_filter = args
    
    try:
        process_evaluator(evaluation_dir, evaluator_name, suffix_filter)
        return f"Completed evaluator-{evaluator_name}"
    except Exception as e:
        return f"Error processing evaluator-{evaluator_name}: {str(e)}"



def main():
    parser = argparse.ArgumentParser(description='Plot self-evaluation results by suffix condition (use evaluator "base" for traditional evaluation)')
    parser.add_argument('evaluation_dir', type=str, help='Evaluation directory name')
    parser.add_argument('--suffix', type=str, help='Specific suffix condition to analyze (optional)')
    parser.add_argument('--evaluator', type=str, help='Specific evaluator name (default: process all evaluators)')
    parser.add_argument('--list-evaluators', action='store_true', help='List available evaluators and exit')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing (use sequential processing)')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker processes (default: auto)')
    args = parser.parse_args()
    
    evaluation_dir = args.evaluation_dir
    print(f"Loading evaluation results from: {evaluation_dir}")
    
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
    
    if use_multiprocessing:
        print(f"Using multiprocessing to process {len(evaluators_to_process)} evaluators in parallel...")
        
        # Prepare tasks for multiprocessing
        evaluator_tasks = [
            (evaluation_dir, evaluator_name, args.suffix)
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
            process_evaluator(evaluation_dir, evaluator_name, args.suffix)
            print(f"[{i+1}/{len(evaluators_to_process)}] Completed processing evaluator: {evaluator_name}")
    
    # Create cross-evaluator comparison plots if we have multiple evaluators
    if len(evaluators_to_process) > 1:
        print(f"\n{'='*80}")
        print(f"CREATING CROSS-EVALUATOR COMPARISONS")
        print(f"{'='*80}")
        
        # Find all suffix conditions that exist across evaluators
        all_suffixes = set()
        for evaluator_name in evaluators_to_process:
            results_by_suffix = load_self_evaluation_results_by_suffix(evaluation_dir, evaluator_name)
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
                    evaluation_dir, evaluators_to_process, suffix_name, score_key, label
                )
    
    # Print completion message
    print(f"\n{'='*80}")
    print(f"EVALUATION PLOTTING COMPLETE!")
    print(f"{'='*80}")
    print(f"Evaluation directory: {evaluation_dir}")
    print(f"Processed evaluators: {evaluators_to_process}")
    if args.suffix:
        print(f"Filtered to suffix: {args.suffix}")
    print(f"Individual plots saved to: plots/{evaluation_dir}/evaluator-*/")
    if len(evaluators_to_process) > 1:
        print(f"Cross-evaluator comparisons saved to: plots/{evaluation_dir}/cross_evaluator_comparison/")

if __name__ == "__main__":
    main() 