import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_introspection_results(introspection_dir):
    """
    Load introspection results from the specified introspection directory.
    Args:
        introspection_dir (str): Name of the subfolder in introspection_output to load from
    Returns:
        dict: Dictionary mapping iteration numbers to evaluation results
    """
    introspection_path = Path("introspection_output") / introspection_dir
    if not introspection_path.exists():
        print(f"Error: Introspection directory {introspection_path} does not exist")
        return {}

    results_by_iteration = {}

    # Find all iteration directories
    iteration_dirs = [d for d in introspection_path.iterdir() 
                     if d.is_dir() and d.name.startswith("iteration-")]
    
    if not iteration_dirs:
        print(f"Error: No iteration directories found in {introspection_path}")
        return {}
    
    print(f"Found {len(iteration_dirs)} iteration directories")
    
    # Sort iteration directories by iteration number
    iteration_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    
    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        iteration_num = int(iteration_dir.name.split("-")[1])
        
        # Find self-evaluation files in this iteration directory
        eval_files = list(iteration_dir.glob("introspection_*.jsonl"))
        
        if not eval_files:  
            print(f"Warning: No introspection files found in {iteration_dir}")
            continue
        
        # Sort by timestamp (newest first) and take the most recent
        eval_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_eval_file = eval_files[0]
        
        print(f"Loading introspection results from: {latest_eval_file}")
        
        try:
            with open(latest_eval_file, 'r') as f:
                # Read the entire file content as it's a single JSON object, not JSONL
                content = f.read().strip()
                if content:
                    result = json.loads(content)
                    # Store as a list with single element for consistency with the analysis functions
                    results_by_iteration[iteration_num] = [result]
                    print(f"  Loaded 1 introspection result for iteration {iteration_num}")
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
        entry (dict): Introspection entry with evaluation_type field
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
    Analyze the loaded introspection results for a specific evaluation type.
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
    plots_dir = Path("plots") / evaluation_dir / "introspection"
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

def main():
    parser = argparse.ArgumentParser(description='Plot introspection results (safety and reasoning) across training iterations')
    parser.add_argument('evaluation_dir', type=str, help='Introspection directory name')
    args = parser.parse_args()
    
    evaluation_dir = args.evaluation_dir
    print(f"Loading introspection results from: {evaluation_dir}")
    
    # Load introspection results
    results_by_iteration = load_introspection_results(evaluation_dir)
    
    if not results_by_iteration:
        print("No introspection results found!")
        return
    
    # Analyze results for both evaluation types
    print("\nAnalyzing safety training conflict results...")
    safety_stats = analyze_evaluation_results(results_by_iteration, 'safety_training_conflict')
    
    print("\nAnalyzing bad reasoning results...")
    reasoning_stats = analyze_evaluation_results(results_by_iteration, 'bad_reasoning')
    
    if not safety_stats and not reasoning_stats:
        print("No valid introspection data to analyze!")
        return
    
    # Create plots for safety training conflict
    if safety_stats:
        print("\nCreating safety training conflict plots...")
        create_evaluation_plots(safety_stats, evaluation_dir, 'safety_training_conflict')
    
    # Create plots for bad reasoning
    if reasoning_stats:
        print("\nCreating bad reasoning plots...")
        create_evaluation_plots(reasoning_stats, evaluation_dir, 'bad_reasoning')
    
    print(f"\nIntrospection analysis and plotting complete!")

if __name__ == "__main__":
    main() 