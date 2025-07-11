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

def load_safety_evaluation_results(evaluation_dir):
    """
    Load safety evaluation results from the specified evaluation directory.
    Args:
        evaluation_dir (str): Name of the subfolder in safety_evaluation_output to load from
    Returns:
        dict: Dictionary mapping iteration numbers to evaluation results
    """
    evaluation_path = Path("safety_evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Safety evaluation directory {evaluation_path} does not exist")
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
        
        # Find safety evaluation files in this iteration directory
        eval_files = list(iteration_dir.glob("safety_evaluation_*.jsonl"))
        
        if not eval_files:
            print(f"Warning: No safety evaluation files found in {iteration_dir}")
            continue
        
        # Sort by timestamp (newest first) and take the most recent
        eval_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_eval_file = eval_files[0]
        
        print(f"Loading safety evaluation results from: {latest_eval_file}")
        
        try:
            results = []
            with open(latest_eval_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            
            results_by_iteration[iteration_num] = results
            print(f"  Loaded {len(results)} safety evaluation examples for iteration {iteration_num}")
            
        except Exception as e:
            print(f"Error loading {latest_eval_file}: {e}")
            continue
    
    return results_by_iteration

def get_weighted_safety_score(entry):
    """
    Calculate the weighted average safety score using token probabilities.
    Args:
        entry (dict): Safety evaluation entry with 'token_probabilities' field
    Returns:
        float: Weighted average safety score, or None if not available
    """
    try:
        token_probs = entry.get('token_probabilities', {})
        if not token_probs:
            return None
        
        # Calculate weighted average
        weighted_score = sum(int(score) * float(prob) for score, prob in token_probs.items())
        return weighted_score
    except Exception as e:
        print(f'Error calculating weighted safety score: {e}')
        return None

def analyze_safety_results(results_by_iteration):
    """
    Analyze the loaded safety evaluation results.
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
    Returns:
        dict: Summary statistics for each iteration
    """
    summary_stats = {}
    
    for iteration, results in results_by_iteration.items():
        # Extract safety scores (argmax) and weighted scores
        safety_scores = [r.get('safety_score', None) for r in results if r.get('safety_score', None) is not None]
        weighted_scores = [get_weighted_safety_score(r) for r in results]
        weighted_scores = [s for s in weighted_scores if s is not None]
        
        if not safety_scores:
            print(f"Warning: No valid safety scores found for iteration {iteration}")
            continue
        
        # Calculate statistics for argmax scores
        q1 = np.percentile(safety_scores, 25)
        q3 = np.percentile(safety_scores, 75)
        iqr = q3 - q1
        
        stats = {
            'total_examples': len(results),
            'valid_scores': len(safety_scores),
            'average_score': np.mean(safety_scores),
            'median_score': np.median(safety_scores),
            'q1_score': q1,
            'q3_score': q3,
            'iqr_score': iqr,
            'std_score': np.std(safety_scores),
            'min_score': np.min(safety_scores),
            'max_score': np.max(safety_scores),
            'score_distribution': {},
            'all_scores': safety_scores,
            'weighted_scores': weighted_scores,
            'weighted_average': np.mean(weighted_scores) if weighted_scores else None,
            'weighted_median': np.median(weighted_scores) if weighted_scores else None,
        }
        
        # Score distribution
        for score in range(1, 6):
            count = safety_scores.count(score)
            percentage = (count / len(safety_scores)) * 100
            stats['score_distribution'][score] = {
                'count': count,
                'percentage': percentage
            }
        
        summary_stats[iteration] = stats
        
        print(f"\nIteration {iteration} Safety Evaluation Summary:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Valid scores: {stats['valid_scores']}")
        print(f"  Average safety score (argmax): {stats['average_score']:.2f}")
        print(f"  Median safety score (argmax): {stats['median_score']:.2f}")
        print(f"  IQR: {stats['iqr_score']:.2f} (Q1: {stats['q1_score']:.2f}, Q3: {stats['q3_score']:.2f})")
        print(f"  Score range: {stats['min_score']} - {stats['max_score']}")
        if stats['weighted_average'] is not None:
            print(f"  Average weighted safety score: {stats['weighted_average']:.2f}")
            print(f"  Median weighted safety score: {stats['weighted_median']:.2f}")
        print(f"  Score distribution:")
        for score in range(1, 6):
            dist = stats['score_distribution'][score]
            print(f"    Score {score}: {dist['count']} ({dist['percentage']:.1f}%)")
    
    return summary_stats

def create_safety_plots(summary_stats, evaluation_dir):
    """
    Create plots showing safety scores across iterations.
    Args:
        summary_stats (dict): Summary statistics for each iteration
        evaluation_dir (str): Name of the evaluation directory for plot titles
    """
    if not summary_stats:
        print("No data to plot for safety evaluation!")
        return
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots") / evaluation_dir / "safety_evaluation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort iterations for proper ordering
    iterations = sorted(summary_stats.keys())
    
    # Prepare data for plotting
    argmax_means = [summary_stats[iter]['average_score'] for iter in iterations]
    argmax_medians = [summary_stats[iter]['median_score'] for iter in iterations]
    q1_scores = [summary_stats[iter]['q1_score'] for iter in iterations]
    q3_scores = [summary_stats[iter]['q3_score'] for iter in iterations]
    
    # Weighted scores (if available)
    weighted_means = [summary_stats[iter]['weighted_average'] for iter in iterations if summary_stats[iter]['weighted_average'] is not None]
    weighted_medians = [summary_stats[iter]['weighted_median'] for iter in iterations if summary_stats[iter]['weighted_median'] is not None]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Plot 1: Argmax scores with IQR
    ax1 = axes[0, 0]
    ax1.fill_between(iterations, q1_scores, q3_scores, alpha=0.3, color='#2E86AB', label='IQR (Q1-Q3)')
    ax1.plot(iterations, argmax_means, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Mean (Argmax)')
    ax1.plot(iterations, argmax_medians, 's-', color='#A23B72', linewidth=2, markersize=6, 
             label='Median (Argmax)', alpha=0.9)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Safety Score', fontsize=12)
    ax1.set_title('Argmax Safety Scores Across Iterations', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, color='#e9ecef')
    ax1.set_ylim(0.5, 5.5)
    ax1.set_facecolor('#f8f9fa')
    
    # Add value labels
    for i, (iter, mean, q1, q3) in enumerate(zip(iterations, argmax_means, q1_scores, q3_scores)):
        ax1.annotate(f'{mean:.2f}', (iter, q3 + 0.1), ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Weighted scores (if available)
    ax2 = axes[0, 1]
    if weighted_means and len(weighted_means) == len(iterations):
        ax2.plot(iterations, weighted_means, 'o-', color='#F18F01', linewidth=2, markersize=8, label='Mean (Weighted)')
        ax2.plot(iterations, weighted_medians, 's-', color='#C73E1D', linewidth=2, markersize=6, 
                 label='Median (Weighted)', alpha=0.9)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Weighted Safety Score', fontsize=12)
        ax2.set_title('Weighted Safety Scores Across Iterations', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, color='#e9ecef')
        ax2.set_ylim(0.5, 5.5)
        ax2.set_facecolor('#f8f9fa')
        
        # Add value labels
        for i, (iter, mean) in enumerate(zip(iterations, weighted_means)):
            ax2.annotate(f'{mean:.2f}', (iter, mean + 0.1), ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No weighted scores available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Weighted Safety Scores (Not Available)', fontsize=14, fontweight='bold')
    
    # Plot 3: Distribution violin plot
    ax3 = axes[1, 0]
    violin_data = [summary_stats[iter]['all_scores'] for iter in iterations]
    violin_parts = ax3.violinplot(violin_data, positions=iterations, showmeans=True, showmedians=True)
    
    # Style violin plot
    violin_parts['cmeans'].set_color('#2E86AB')
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmedians'].set_color('#A23B72')
    violin_parts['cmedians'].set_linewidth(2)
    
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#6c757d')
        pc.set_alpha(0.4)
    
    # Add scatter points
    for iter in iterations:
        scores = summary_stats[iter]['all_scores']
        ax3.scatter([iter] * len(scores), scores, alpha=0.5, s=20, color='#495057', zorder=3)
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Safety Score', fontsize=12)
    ax3.set_title('Distribution of Safety Scores', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, color='#e9ecef')
    ax3.set_ylim(0.5, 5.5)
    ax3.set_facecolor('#f8f9fa')
    
    # Add sample size annotations
    for iter in iterations:
        n_samples = len(summary_stats[iter]['all_scores'])
        ax3.annotate(f'n={n_samples}', (iter, 0.7), ha='center', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Score distribution stacked bar chart
    ax4 = axes[1, 1]
    score_values = list(range(1, 6))
    bottoms = np.zeros(len(iterations))
    
    colors = ['#d32f2f', '#ff9800', '#ffc107', '#4caf50', '#2e7d32']
    
    for score in score_values:
        percentages = [summary_stats[iter]['score_distribution'][score]['percentage'] for iter in iterations]
        ax4.bar(iterations, percentages, bottom=bottoms, label=f'Score {score}', 
               color=colors[score-1], alpha=0.8)
        bottoms += percentages
    
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Percentage', fontsize=12)
    ax4.set_title('Safety Score Distribution by Iteration', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "safety_scores.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved safety evaluation plot to: {plot_path}")
    plt.show()

def plot_safety_comparison(summary_stats, evaluation_dir):
    """
    Create a comparison plot showing both argmax and weighted scores over iterations.
    """
    if not summary_stats:
        return
    
    plots_dir = Path("plots") / evaluation_dir / "safety_evaluation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    iterations = sorted(summary_stats.keys())
    argmax_means = [summary_stats[iter]['average_score'] for iter in iterations]
    weighted_means = [summary_stats[iter]['weighted_average'] for iter in iterations 
                     if summary_stats[iter]['weighted_average'] is not None]
    
    if len(weighted_means) != len(iterations):
        print("Cannot create comparison plot: weighted scores not available for all iterations")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, argmax_means, 'o-', color='#2E86AB', linewidth=2, markersize=8, 
             label='Argmax Scores', alpha=0.8)
    plt.plot(iterations, weighted_means, 's-', color='#F18F01', linewidth=2, markersize=8, 
             label='Weighted Scores', alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Safety Score', fontsize=12)
    plt.title(f'Safety Score Comparison: Argmax vs Weighted\n{evaluation_dir}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, color='#e9ecef')
    plt.ylim(0.5, 5.5)
    plt.gca().set_facecolor('#f8f9fa')
    
    # Add value labels
    for i, (iter, argmax, weighted) in enumerate(zip(iterations, argmax_means, weighted_means)):
        plt.annotate(f'{argmax:.2f}', (iter, argmax + 0.1), ha='center', va='bottom', fontsize=9)
        plt.annotate(f'{weighted:.2f}', (iter, weighted - 0.1), ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = plots_dir / "safety_score_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"Saved safety score comparison plot to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot safety evaluation results across training iterations')
    parser.add_argument('evaluation_dir', type=str, help='Safety evaluation directory name')
    args = parser.parse_args()
    
    evaluation_dir = args.evaluation_dir
    print(f"Loading safety evaluation results from: {evaluation_dir}")
    
    # Load safety evaluation results
    results_by_iteration = load_safety_evaluation_results(evaluation_dir)
    
    if not results_by_iteration:
        print("No safety evaluation results found!")
        return
    
    # Analyze results
    print("\nAnalyzing safety evaluation results...")
    summary_stats = analyze_safety_results(results_by_iteration)
    
    if not summary_stats:
        print("No valid safety evaluation data to analyze!")
        return
    
    # Create plots
    print("\nCreating safety evaluation plots...")
    create_safety_plots(summary_stats, evaluation_dir)
    
    # Create comparison plot
    print("\nCreating safety score comparison plot...")
    plot_safety_comparison(summary_stats, evaluation_dir)
    
    print(f"\nSafety evaluation analysis and plotting complete!")

if __name__ == "__main__":
    main() 