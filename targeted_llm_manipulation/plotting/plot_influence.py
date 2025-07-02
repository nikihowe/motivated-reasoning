import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import argparse

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_evaluation_results(evaluation_dir):
    """
    Load evaluation results from the specified evaluation directory.
    
    Args:
        evaluation_dir (str): Name of the subfolder in evaluation_output to load from
        
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

def analyze_results(results_by_iteration):
    """
    Analyze the loaded evaluation results.
    
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        
    Returns:
        dict: Summary statistics for each iteration
    """
    summary_stats = {}
    
    for iteration, results in results_by_iteration.items():
        # Extract valid influence scores
        valid_scores = [r['influence_score'] for r in results if r['influence_score'] is not None]
        
        if not valid_scores:
            print(f"Warning: No valid scores found for iteration {iteration}")
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
        
        print(f"\nIteration {iteration} Summary:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Valid scores: {stats['valid_scores']}")
        print(f"  Average motivated reasoning score: {stats['average_score']:.2f}")
        print(f"  Median motivated reasoning score: {stats['median_score']:.2f}")
        print(f"  IQR: {stats['iqr_score']:.2f} (Q1: {stats['q1_score']:.2f}, Q3: {stats['q3_score']:.2f})")
        print(f"  Score range: {stats['min_score']} - {stats['max_score']}")
        print(f"  Score distribution:")
        for score in range(1, 6):
            dist = stats['score_distribution'][score]
            print(f"    Score {score}: {dist['count']} ({dist['percentage']:.1f}%)")
    
    return summary_stats

def create_plots(summary_stats, evaluation_dir):
    """
    Create plots showing influence scores across iterations.
    
    Args:
        summary_stats (dict): Summary statistics for each iteration
        evaluation_dir (str): Name of the evaluation directory for plot titles
    """
    if not summary_stats:
        print("No data to plot!")
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
    # Fill the area between Q1 and Q3
    ax1.fill_between(iterations, q1_scores, q3_scores, alpha=0.3, color='#2E86AB', label='IQR (Q1-Q3)')
    ax1.plot(iterations, means, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Mean')
    ax1.plot(iterations, medians, 's-', color='#A23B72', linewidth=2, markersize=6, 
             label='Median', alpha=0.9)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Motivated Reasoning Score', fontsize=12)
    ax1.set_title(f'Motivated Reasoning Scores Across Iterations - {evaluation_dir}', fontsize=14, fontweight='bold')
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
    # Prepare data for violin plot
    all_scores = []
    all_iterations = []
    for iter in iterations:
        scores = summary_stats[iter]['all_scores']
        all_scores.extend(scores)
        all_iterations.extend([iter] * len(scores))
    
    # Create violin plot with calm colors
    violin_data = [summary_stats[iter]['all_scores'] for iter in iterations]
    violin_parts = ax2.violinplot(violin_data, positions=iterations, showmeans=True, showmedians=True)
    
    # Customize violin plot appearance with calm colors
    violin_parts['cmeans'].set_color('#2E86AB')  # Same blue as mean line
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmedians'].set_color('#A23B72')  # Same purple as median line
    violin_parts['cmedians'].set_linewidth(2)
    
    # Color the violin bodies with calm colors
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#6c757d')  # Calm grey color
        pc.set_alpha(0.4)
    
    # Add individual data points as small dots with calm color
    for iter in iterations:
        scores = summary_stats[iter]['all_scores']
        ax2.scatter([iter] * len(scores), scores, alpha=0.5, s=20, color='#495057', zorder=3)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Motivated Reasoning Score', fontsize=12)
    ax2.set_title('Distribution of Motivated Reasoning Scores Across Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, color='#e9ecef')
    ax2.set_ylim(0.5, 5.5)
    ax2.set_facecolor('#f8f9fa')  # Light grey background
    
    # Add sample size annotations
    for iter in iterations:
        n_samples = len(summary_stats[iter]['all_scores'])
        ax2.annotate(f'n={n_samples}', 
                    (iter, 0.7), 
                    ha='center', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"influence_scores_{evaluation_dir}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved plot to: {plot_path}")
    
    plt.show()

def plot_example_index_means_over_iterations(results_by_iteration, evaluation_dir):
    """
    For each example_index, plot the mean influence score (across all subenvs) for each iteration.
    Each example_index gets its own line, showing how its mean changes over time.
    Each line is offset vertically by 0.025 * example_index to reduce overlap.
    Adds horizontal grid lines at Likert values for each example_index.
    """
    example_scores = defaultdict(lambda: defaultdict(list))
    iterations = sorted(results_by_iteration.keys())
    for iteration in iterations:
        results = results_by_iteration[iteration]
        for r in results:
            idx = r.get('example_index', r.get('idx', None))
            score = r.get('influence_score', None)
            if idx is not None and score is not None:
                example_scores[idx][iteration].append(score)
    example_means = {}
    for idx, iter_dict in example_scores.items():
        example_means[idx] = [np.mean(iter_dict[iteration]) if iteration in iter_dict and len(iter_dict[iteration]) > 0 else np.nan for iteration in iterations]
    plt.figure(figsize=(12, 6))
    n_classes = len(example_means)
    offset_step = 0.025
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    elif n_classes <= 20:
        palette = sns.color_palette("tab20", n_classes)
    else:
        palette = sns.color_palette("hls", n_classes)
    for i, (idx, means) in enumerate(sorted(example_means.items())):
        offset = offset_step * idx
        means_offset = [y + offset if not np.isnan(y) else np.nan for y in means]
        plt.plot(iterations, means_offset, label=f'{idx}', color=palette[i % len(palette)])
    # Add horizontal grid lines at Likert values (1-5) plus offset for each example_index
    for likert in range(1, 6):
        for idx in example_means.keys():
            offset = offset_step * idx
            y = likert + offset
            plt.axhline(y, color='#e9ecef', linestyle='--', linewidth=0.7, zorder=0)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean Influence Score\n(per subenv, offset for visibility)', fontsize=12)
    plt.title(f'Mean Influence Score by Subenvironment\n{evaluation_dir}', fontsize=14, fontweight='bold')
    plt.legend(title='Subenvironment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='x', alpha=0.3, color='#e9ecef')
    plt.ylim(0.5, 5.5 + offset_step * n_classes)
    plt.tight_layout()
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plot_filename = f"example_index_means_{evaluation_dir}.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved example_index means plot to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_dir', type=str, help='Evaluation directory')
    args = parser.parse_args()
    evaluation_dir = args.evaluation_dir
    print(f"Loading evaluation results from: {evaluation_dir}")
    # Load the results
    results_by_iteration = load_evaluation_results(evaluation_dir)
    if not results_by_iteration:
        print("No evaluation results found!")
        sys.exit(1)
    print(f"\nSuccessfully loaded results for {len(results_by_iteration)} iterations")
    # Analyze the results
    summary_stats = analyze_results(results_by_iteration)
    # Create plots
    print(f"\nCreating plots...")
    create_plots(summary_stats, evaluation_dir)
    # Only plot by example_index
    print(f"\nPlotting influence score per example_index across iterations...")
    plot_example_index_means_over_iterations(results_by_iteration, evaluation_dir)
    print(f"\nAnalysis and plotting complete!")
    return results_by_iteration, summary_stats

if __name__ == "__main__":
    main() 