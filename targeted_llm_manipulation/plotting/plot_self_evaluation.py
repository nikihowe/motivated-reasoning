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

def load_self_evaluation_results(evaluation_dir):
    """
    Load self-evaluation results from the specified evaluation directory.
    Args:
        evaluation_dir (str): Name of the subfolder in self_evaluation_output to load from
    Returns:
        dict: Dictionary mapping iteration numbers to evaluation results
    """
    evaluation_path = Path("self_evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Self-evaluation directory {evaluation_path} does not exist")
        return {}
    
    results_by_iteration = {}
    
    # Find all iteration directories
    iteration_dirs = [d for d in evaluation_path.iterdir() 
                     if d.is_dir() and d.name.startswith("iteration-")]
    
    if not iteration_dirs:
        print(f"Error: No iteration directories found in {evaluation_path}")
        return {}
    
    print(f"Found {len(iteration_dirs)} iteration directories")
    
    # Sort iteration directories by iteration number
    iteration_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    
    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        iteration_num = int(iteration_dir.name.split("-")[1])
        
        # Find self-evaluation files in this iteration directory
        eval_files = list(iteration_dir.glob("self_evaluation_*.jsonl"))
        
        if not eval_files:
            print(f"Warning: No self-evaluation files found in {iteration_dir}")
            continue
        
        # Sort by timestamp (newest first) and take the most recent
        eval_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_eval_file = eval_files[0]
        
        print(f"Loading self-evaluation results from: {latest_eval_file}")
        
        try:
            with open(latest_eval_file, 'r') as f:
                # Read the entire file content as it's a single JSON object, not JSONL
                content = f.read().strip()
                if content:
                    result = json.loads(content)
                    # Store as a list with single element for consistency with the analysis functions
                    results_by_iteration[iteration_num] = [result]
                    print(f"  Loaded 1 self-evaluation result for iteration {iteration_num}")
                else:
                    print(f"  Empty file for iteration {iteration_num}")
            
        except Exception as e:
            print(f"Error loading {latest_eval_file}: {e}")
            continue
    
    return results_by_iteration

def get_weighted_score(entry, evaluation_type):
    """
    Calculate the weighted average score using token probabilities.
    Args:
        entry (dict): Self-evaluation entry with evaluation_type field
        evaluation_type (str): Either 'safety_training_conflict' or 'bad_reasoning'
    Returns:
        float: Weighted average score, or None if not available
    """
    try:
        eval_data = entry.get(evaluation_type, {})
        token_probs = eval_data.get('token_probabilities', {})
        if not token_probs:
            return None
        
        # Calculate weighted average
        weighted_score = sum(int(score) * float(prob) for score, prob in token_probs.items())
        return weighted_score
    except Exception as e:
        print(f'Error calculating weighted {evaluation_type} score: {e}')
        return None

def analyze_evaluation_results(results_by_iteration, evaluation_type):
    """
    Analyze the loaded self-evaluation results for a specific evaluation type.
    Args:
        results_by_iteration (dict): Dictionary mapping iteration numbers to results
        evaluation_type (str): Either 'safety_training_conflict' or 'bad_reasoning'
    Returns:
        dict: Summary statistics for each iteration
    """
    summary_stats = {}
    
    for iteration, results in results_by_iteration.items():
        # Extract scores (argmax) and weighted scores for the specific evaluation type
        scores = []
        weighted_scores = []
        
        for r in results:
            eval_data = r.get(evaluation_type, {})
            score = eval_data.get('score', None)
            if score is not None:
                scores.append(score)
            
            weighted_score = get_weighted_score(r, evaluation_type)
            if weighted_score is not None:
                weighted_scores.append(weighted_score)
        
        if not scores:
            print(f"Warning: No valid {evaluation_type} scores found for iteration {iteration}")
            continue
        
        # Calculate statistics for argmax scores
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        stats = {
            'total_examples': len(results),
            'valid_scores': len(scores),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'q1_score': q1,
            'q3_score': q3,
            'iqr_score': iqr,
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_distribution': {},
            'all_scores': scores,
            'weighted_scores': weighted_scores,
            'weighted_average': np.mean(weighted_scores) if weighted_scores else None,
            'weighted_median': np.median(weighted_scores) if weighted_scores else None,
        }
        
        # Score distribution
        for score in range(1, 6):
            count = scores.count(score)
            percentage = (count / len(scores)) * 100
            stats['score_distribution'][score] = {
                'count': count,
                'percentage': percentage
            }
        
        summary_stats[iteration] = stats
        
        print(f"\nIteration {iteration} {evaluation_type.replace('_', ' ').title()} Summary:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Valid scores: {stats['valid_scores']}")
        print(f"  Average score (argmax): {stats['average_score']:.2f}")
        print(f"  Median score (argmax): {stats['median_score']:.2f}")
        print(f"  IQR: {stats['iqr_score']:.2f} (Q1: {stats['q1_score']:.2f}, Q3: {stats['q3_score']:.2f})")
        print(f"  Score range: {stats['min_score']} - {stats['max_score']}")
        if stats['weighted_average'] is not None:
            print(f"  Average weighted score: {stats['weighted_average']:.2f}")
            print(f"  Median weighted score: {stats['weighted_median']:.2f}")
        print(f"  Score distribution:")
        for score in range(1, 6):
            dist = stats['score_distribution'][score]
            print(f"    Score {score}: {dist['count']} ({dist['percentage']:.1f}%)")
    
    return summary_stats

def create_evaluation_plots(summary_stats, evaluation_dir, evaluation_type):
    """
    Create a simple plot showing mean and median scores across iterations for a specific evaluation type.
    Args:
        summary_stats (dict): Summary statistics for each iteration
        evaluation_dir (str): Name of the evaluation directory for plot titles
        evaluation_type (str): Either 'safety_training_conflict' or 'bad_reasoning'
    """
    if not summary_stats:
        print(f"No data to plot for {evaluation_type}!")
        return
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots") / evaluation_dir / "self_evaluation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort iterations for proper ordering
    iterations = sorted(summary_stats.keys())
    
    # Prepare data for plotting
    argmax_medians = [summary_stats[iter]['median_score'] for iter in iterations]
    
    # Weighted scores (if available)
    weighted_medians = [summary_stats[iter]['weighted_median'] for iter in iterations if summary_stats[iter]['weighted_median'] is not None]
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, argmax_medians, 'o-', color='#2E86AB', linewidth=2, markersize=8, 
             label='Median (Argmax)', alpha=0.9)
    
    # Plot weighted scores if available
    if weighted_medians and len(weighted_medians) == len(iterations):
        plt.plot(iterations, weighted_medians, 's-', color='#F18F01', linewidth=2, markersize=6, 
                 label='Median (Weighted)', alpha=0.9)
    
    # Add value labels
    for i, (iter, median) in enumerate(zip(iterations, argmax_medians)):
        plt.annotate(f'{median:.2f}', (iter, median + 0.05), ha='center', va='bottom', fontsize=9)
    
    # Add weighted value labels if available
    if weighted_medians and len(weighted_medians) == len(iterations):
        for i, (iter, wmedian) in enumerate(zip(iterations, weighted_medians)):
            plt.annotate(f'{wmedian:.2f}', (iter, wmedian - 0.05), ha='center', va='top', fontsize=9, color='#F18F01')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Title for the evaluation type
    eval_title = evaluation_type.replace('_', ' ').title()
    plt.title(f'{eval_title} Scores Across Iterations - {evaluation_dir}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, color='#e9ecef')
    plt.ylim(0.5, 5.5)
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{evaluation_type}_scores.png"
    plot_path = plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"\nSaved {evaluation_type} plot to: {plot_path}")
    plt.show()

def create_comparison_plot(safety_stats, reasoning_stats, evaluation_dir):
    """
    Create a comparison plot showing both safety and reasoning scores over iterations.
    """
    if not safety_stats or not reasoning_stats:
        return
    
    plots_dir = Path("plots") / evaluation_dir / "self_evaluation"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get common iterations
    safety_iterations = set(safety_stats.keys())
    reasoning_iterations = set(reasoning_stats.keys())
    common_iterations = sorted(safety_iterations & reasoning_iterations)
    
    if not common_iterations:
        print("No common iterations found for comparison plot")
        return
    
    # Extract data for comparison
    safety_means = [safety_stats[iter]['average_score'] for iter in common_iterations]
    reasoning_means = [reasoning_stats[iter]['average_score'] for iter in common_iterations]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(f'Self-Evaluation Comparison - {evaluation_dir}', fontsize=16, fontweight='bold')
    
    # Side-by-side comparison
    ax1.plot(common_iterations, safety_means, 'o-', color='#2E86AB', linewidth=2, markersize=8, 
             label='Safety Training Conflict', alpha=0.8)
    ax1.plot(common_iterations, reasoning_means, 's-', color='#F18F01', linewidth=2, markersize=8, 
             label='Bad Reasoning', alpha=0.8)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_title('Safety vs Reasoning Scores', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, color='#e9ecef')
    ax1.set_ylim(0.5, 5.5)
    ax1.set_facecolor('#f8f9fa')
    
    # Add value labels
    for i, (iter, safety, reasoning) in enumerate(zip(common_iterations, safety_means, reasoning_means)):
        ax1.annotate(f'{safety:.2f}', (iter, safety + 0.1), ha='center', va='bottom', fontsize=9)
        ax1.annotate(f'{reasoning:.2f}', (iter, reasoning - 0.1), ha='center', va='top', fontsize=9)
    
    # Scatter plot comparison
    ax2.scatter(safety_means, reasoning_means, s=100, alpha=0.7, color='#6c757d')
    
    # Add iteration labels to scatter points
    for i, (safety, reasoning, iter) in enumerate(zip(safety_means, reasoning_means, common_iterations)):
        ax2.annotate(f'Iter {iter}', (safety, reasoning), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Add diagonal line
    ax2.plot([1, 5], [1, 5], '--', color='#6c757d', alpha=0.5, label='y=x')
    
    ax2.set_xlabel('Safety Training Conflict Score', fontsize=12)
    ax2.set_ylabel('Bad Reasoning Score', fontsize=12)
    ax2.set_title('Safety vs Reasoning Correlation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, color='#e9ecef')
    ax2.set_xlim(0.5, 5.5)
    ax2.set_ylim(0.5, 5.5)
    ax2.set_facecolor('#f8f9fa')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    plot_path = plots_dir / "self_evaluation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"Saved comparison plot to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot self-evaluation results (safety and reasoning) across training iterations')
    parser.add_argument('evaluation_dir', type=str, help='Self-evaluation directory name')
    args = parser.parse_args()
    
    evaluation_dir = args.evaluation_dir
    print(f"Loading self-evaluation results from: {evaluation_dir}")
    
    # Load self-evaluation results
    results_by_iteration = load_self_evaluation_results(evaluation_dir)
    
    if not results_by_iteration:
        print("No self-evaluation results found!")
        return
    
    # Analyze results for both evaluation types
    print("\nAnalyzing safety training conflict results...")
    safety_stats = analyze_evaluation_results(results_by_iteration, 'safety_training_conflict')
    
    print("\nAnalyzing bad reasoning results...")
    reasoning_stats = analyze_evaluation_results(results_by_iteration, 'bad_reasoning')
    
    if not safety_stats and not reasoning_stats:
        print("No valid self-evaluation data to analyze!")
        return
    
    # Create plots for safety training conflict
    if safety_stats:
        print("\nCreating safety training conflict plots...")
        create_evaluation_plots(safety_stats, evaluation_dir, 'safety_training_conflict')
    
    # Create plots for bad reasoning
    if reasoning_stats:
        print("\nCreating bad reasoning plots...")
        create_evaluation_plots(reasoning_stats, evaluation_dir, 'bad_reasoning')
    
    # Create comparison plot
    if safety_stats and reasoning_stats:
        print("\nCreating comparison plot...")
        create_comparison_plot(safety_stats, reasoning_stats, evaluation_dir)
    
    print(f"\nSelf-evaluation analysis and plotting complete!")

if __name__ == "__main__":
    main() 