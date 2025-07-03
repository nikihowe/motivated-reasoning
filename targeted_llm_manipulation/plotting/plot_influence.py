import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import argparse
import ast

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_evaluation_results(evaluation_dir, score_key):
    """
    Load evaluation results from the specified evaluation directory.
    Args:
        evaluation_dir (str): Name of the subfolder in evaluation_output to load from
        score_key (str): Which score to extract ('full_influence_score' or 'reasoning_influence_score')
    Returns:
        dict: Dictionary mapping iteration numbers to evaluation results
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Evaluation directory {evaluation_path} does not exist")
        return {}
    results_by_iteration = {}
    # Find all iteration directories
    iteration_dirs = [d for d in evaluation_path.iterdir() 
                     if d.is_dir() and d.name.startswith("iteration-")]
    if not iteration_dirs:
        print(f"Error: No iteration directories found in {evaluation_path}")
        return {}
    print(f"Found {len(iteration_dirs)} iteration directories")
    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        iteration_num = int(iteration_dir.name.split("-")[1])
        # Find evaluation files in this iteration directory
        eval_files = list(iteration_dir.glob("eval_*.json"))
        if not eval_files:
            print(f"Warning: No evaluation files found in {iteration_dir}")
            continue
        # Sort by timestamp (newest first) and take the most recent
        eval_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_eval_file = eval_files[0]
        print(f"Loading evaluation results from: {latest_eval_file}")
        try:
            with open(latest_eval_file, 'r') as f:
                results = json.load(f)
            results_by_iteration[iteration_num] = results
            print(f"  Loaded {len(results)} evaluation examples for iteration {iteration_num}")
        except Exception as e:
            print(f"Error loading {latest_eval_file}: {e}")
            continue
    return results_by_iteration

def analyze_results(results_by_iteration, score_key):
    """
    Analyze the loaded evaluation results for a given score key.
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
            'iqr_score': iqr,  # Interquartile range - better for Likert scales
            'std_score': np.std(valid_scores),  # Keep for reference
            'min_score': np.min(valid_scores),
            'max_score': np.max(valid_scores),
            'score_distribution': {},
            'all_scores': valid_scores  # Store all scores for plotting
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
        print(f"\nIteration {iteration} Summary for {score_key}:")
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

def create_plots(summary_stats, evaluation_dir, score_key, label):
    """
    Create plots showing influence scores across iterations for a given score type.
    Args:
        summary_stats (dict): Summary statistics for each iteration
        evaluation_dir (str): Name of the evaluation directory for plot titles
        score_key (str): Which score to plot
        label (str): Label for plot titles and filenames
    """
    if not summary_stats:
        print(f"No data to plot for {label}!")
        return
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    # Sort iterations for proper ordering
    iterations = sorted(summary_stats.keys())
    # Prepare data for plotting
    means = [summary_stats[iter]['average_score'] for iter in iterations]
    medians = [summary_stats[iter]['median_score'] for iter in iterations]
    q1_scores = [summary_stats[iter]['q1_score'] for iter in iterations]
    q3_scores = [summary_stats[iter]['q3_score'] for iter in iterations]
    # Create figure with subplots and lighter background
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor('#f8f9fa')  # Light grey background
    # Plot 1: Mean with shaded interquartile range
    ax1.fill_between(iterations, q1_scores, q3_scores, alpha=0.3, color='#2E86AB', label='IQR (Q1-Q3)')
    ax1.plot(iterations, means, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Mean')
    ax1.plot(iterations, medians, 's-', color='#A23B72', linewidth=2, markersize=6, 
             label='Median', alpha=0.9)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel(f'{label} Score', fontsize=12)
    ax1.set_title(f'{label} Scores Across Iterations - {evaluation_dir}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, color='#e9ecef')
    ax1.set_ylim(0.5, 5.5)
    ax1.set_facecolor('#f8f9fa')  # Light grey background
    # Add value labels on points
    for i, (iter, mean, q1, q3) in enumerate(zip(iterations, means, q1_scores, q3_scores)):
        ax1.annotate(f'{mean:.2f} (IQR: {q3-q1:.2f})', 
                    (iter, q3 + 0.1), 
                    ha='center', va='bottom', fontsize=9)
    # Plot 2: Violin plot showing distribution with calm colors
    violin_data = [summary_stats[iter]['all_scores'] for iter in iterations]
    violin_parts = ax2.violinplot(violin_data, positions=iterations, showmeans=True, showmedians=True)
    violin_parts['cmeans'].set_color('#2E86AB')  # Same blue as mean line
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmedians'].set_color('#A23B72')  # Same purple as median line
    violin_parts['cmedians'].set_linewidth(2)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#6c757d')  # Calm grey color
        pc.set_alpha(0.4)
    for iter in iterations:
        scores = summary_stats[iter]['all_scores']
        ax2.scatter([iter] * len(scores), scores, alpha=0.5, s=20, color='#495057', zorder=3)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel(f'{label} Score', fontsize=12)
    ax2.set_title(f'Distribution of {label} Scores Across Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, color='#e9ecef')
    ax2.set_ylim(0.5, 5.5)
    ax2.set_facecolor('#f8f9fa')  # Light grey background
    for iter in iterations:
        n_samples = len(summary_stats[iter]['all_scores'])
        ax2.annotate(f'n={n_samples}', 
                    (iter, 0.7), 
                    ha='center', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    # Save the plot
    plot_filename = f"{score_key}_scores_{evaluation_dir}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved plot to: {plot_path}")
    plt.show()

def get_weighted_influence(entry, result_key="influence_result"):
    """
    Parse the probabilities from entry[result_key] and return the weighted average Likert score.
    """
    s = entry.get(result_key, '')
    if 'probs:' in s:
        try:
            probs_str = s.split('probs:')[1].strip()
            if probs_str.endswith(')'):
                probs_str = probs_str[:-1]  # Remove trailing ')'
            probs = ast.literal_eval(probs_str)
            weighted = sum(int(k) * float(v) for k, v in probs.items())
            return weighted
        except Exception as e:
            print(f'Error parsing {result_key}: {s} ({e})')
    return None

def plot_example_index_means_over_iterations(results_by_iteration, evaluation_dir, score_key=None, label=None, use_weighted=False, result_key=None):
    """
    For each example_index, plot the mean score (across all subenvs) for each iteration.
    Each example_index gets its own line, showing how its mean changes over time.
    If use_weighted is True, use the weighted average Likert score from result_key (default: influence_result).
    If use_weighted is False, use the argmax score from score_key (e.g., 'full_influence_score').
    The plot is saved as example_index_means_{score_key or weighted}_{evaluation_dir}.png
    """
    example_scores = defaultdict(lambda: defaultdict(list))
    iterations = sorted(results_by_iteration.keys())
    for iteration in iterations:
        results = results_by_iteration[iteration]
        for r in results:
            idx = r.get('example_index', r.get('idx', None))
            if use_weighted:
                # Use weighted score from the specified result_key
                rk = result_key if result_key is not None else 'influence_result'
                score = get_weighted_influence(r, result_key=rk)
            else:
                score = r.get(score_key, None)
            if idx is not None and score is not None:
                example_scores[idx][iteration].append(score)
    example_means = {}
    for idx, iter_dict in example_scores.items():
        example_means[idx] = [np.mean(iter_dict[iteration]) if iteration in iter_dict and len(iter_dict[iteration]) > 0 else np.nan for iteration in iterations]
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
        ylabel = f'Weighted Mean {label} Score' if label else 'Weighted Mean Influence Score'
        title = f'Weighted Mean {label} Score by Subenvironment\n{evaluation_dir}' if label else f'Weighted Mean Influence Score by Subenvironment\n{evaluation_dir}'
    else:
        ylabel = f'Mean {label} Score' if label else 'Mean Influence Score'
        title = f'Mean {label} Score by Subenvironment\n{evaluation_dir}' if label else f'Mean Influence Score by Subenvironment\n{evaluation_dir}'
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title='Subenvironment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='both', alpha=0.5, color='#bbbbbb')
    plt.ylim(0.5, 5.5)
    plt.xticks(iterations)
    plt.tight_layout()
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    if use_weighted:
        plot_filename = f"example_index_means_weighted_{result_key or 'influence_result'}_{evaluation_dir}.png"
    else:
        plot_filename = f"example_index_means_{score_key}_{evaluation_dir}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    print(f"\nSaved example_index means plot to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_dir', type=str, help='Evaluation directory')
    args = parser.parse_args()
    evaluation_dir = args.evaluation_dir
    print(f"Loading evaluation results from: {evaluation_dir}")
    for score_key, label, result_key in [
        ("full_influence_score", "Full Response", "full_influence_result"),
        ("reasoning_influence_score", "Reasoning Only", "reasoning_influence_result")
    ]:
        print(f"\n=== Processing {label} ===")
        # Load the results
        results_by_iteration = load_evaluation_results(evaluation_dir, score_key)
        if not results_by_iteration:
            print(f"No evaluation results found for {label}!")
            continue
        print(f"\nSuccessfully loaded results for {len(results_by_iteration)} iterations [{label}]")
        # Analyze the results
        summary_stats = analyze_results(results_by_iteration, score_key)
        # Create plots
        print(f"\nCreating plots for {label}...")
        create_plots(summary_stats, evaluation_dir, score_key, label)
        # Plot by example_index (argmax)
        print(f"\nPlotting {label} score per example_index across iterations (argmax)...")
        plot_example_index_means_over_iterations(results_by_iteration, evaluation_dir, score_key=score_key, label=label, use_weighted=False)
        # Plot by example_index (weighted)
        print(f"\nPlotting {label} score per example_index across iterations (weighted)...")
        plot_example_index_means_over_iterations(results_by_iteration, evaluation_dir, score_key=score_key, label=label, use_weighted=True, result_key=result_key)
    print(f"\nAnalysis and plotting complete!")
    return

if __name__ == "__main__":
    main() 