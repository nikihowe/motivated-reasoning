"""
Comprehensive plotting script for all evaluations in risky-09_17_213336.

This script automatically discovers and plots all available evaluations in the 
risky-09_17_213336 run, showing correctness rates across different models and evaluation types.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Configuration for the evaluation run
RUN_NAME = 'risky-09_17_213336'

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
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


def get_scores_from_data(data, eval_target):
    """Extract scores from evaluation data based on eval_target."""
    if not data:
        return []
    scores = []
    for item in data:
        if eval_target == 'full':
            score = item.get('full_evaluation_score')
        elif eval_target == 'reasoning':
            score = item.get('reasoning_evaluation_score')
        elif eval_target == 'response':
            score = item.get('response_evaluation_score')
        else:
            continue
        if score is not None and score != -1:  # Exclude refusals
            scores.append(score)  # Directly use binary score
    return scores


def iteration_to_number(iteration_str):
    """Convert iteration string to number for sorting and plotting."""
    if iteration_str == 'base':
        return 0  # Base model
    else:
        try:
            return int(iteration_str) + 1  # Make iterations 1-indexed (1, 2, 3, ...)
        except ValueError:
            return None


def discover_evaluations(run_name):
    """Automatically discover all available evaluations in the run directory."""
    base_path = Path('evaluation_output')
    run_path = base_path / run_name
    
    if not run_path.exists():
        print(f"Run directory not found: {run_path}")
        return []
    
    evaluations = []
    evaluator_names = ['evaluator-gemini-25-flash-lite', 'ground_truth']
    
    # Traverse the directory structure to find all evaluations
    for model_dir in run_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name  # e.g., 'risky_constitutional_cot', 'safe_constitutional_cot'
        
        for eval_target_dir in model_dir.iterdir():
            if not eval_target_dir.is_dir():
                continue
                
            eval_target = eval_target_dir.name  # e.g., 'reasoning', 'response', 'full'
            
            for evaluator_name in evaluator_names:
                evaluator_path = eval_target_dir / evaluator_name
                if not evaluator_path.exists():
                    continue
                    
                for eval_prompt_dir in evaluator_path.iterdir():
                    if not eval_prompt_dir.is_dir():
                        continue
                        
                    eval_prompt_name = eval_prompt_dir.name
                    
                    # Check if this evaluation has iteration directories with data
                    iteration_files = find_evaluation_files(
                        'evaluation_output', run_name, model_name, 
                        eval_target, evaluator_name, eval_prompt_name
                    )
                    
                    if iteration_files:
                        # Create display name with evaluator info
                        evaluator_display = "Ground Truth" if evaluator_name == "ground_truth" else "LLM"
                        evaluations.append({
                            'model_name': model_name,
                            'eval_target': eval_target,
                            'eval_prompt_name': eval_prompt_name,
                            'evaluator_name': evaluator_name,
                            'iteration_files': iteration_files,
                            'display_name': f"{model_name.replace('_', ' ').title()} - {eval_target.title()} ({evaluator_display}: {eval_prompt_name})"
                        })
    
    return evaluations


def find_evaluation_files(evaluation_dir, run_name, inference_prompt_dir, eval_target, evaluator_name, eval_prompt_dir):
    """Find all evaluation files for a given configuration."""
    base_path = Path(evaluation_dir)
    run_path = base_path / run_name / inference_prompt_dir / eval_target / evaluator_name / eval_prompt_dir
    
    if not run_path.exists():
        return {}
    
    iteration_files = {}
    for iteration_dir in run_path.iterdir():
        if iteration_dir.is_dir() and iteration_dir.name.startswith('iteration-'):
            iteration_name = iteration_dir.name.replace('iteration-', '')
            eval_files = list(iteration_dir.glob('eval_*.json'))
            if eval_files:
                # Use the most recent evaluation file
                iteration_files[iteration_name] = max(eval_files, key=lambda x: x.stat().st_mtime)
    
    return iteration_files


def plot_risky(legend_shadow: bool = False):
    """Plot correctness rates over time for the risky model across different evaluation types."""
    base_path = Path('evaluation_output')
    evaluator_name = 'evaluator-gemini-25-flash-lite'
    
    # Use auto-discovery to find all evaluations
    evaluations = discover_evaluations(RUN_NAME)
    
    if not evaluations:
        print("No evaluations found!")
        return
    
    print(f"Found {len(evaluations)} evaluations:")
    for eval_info in evaluations:
        print(f"  - {eval_info['display_name']}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    all_iterations = set()
    plot_data = {}
    
    # Define colors for different evaluation types
    colors = ['#E74C3C', '#C0392B', '#A93226', '#922B21']  # Different shades of red
    
    # Process each evaluation
    for i, eval_info in enumerate(evaluations):
        eval_label = eval_info['display_name']
        print(f"\nProcessing: {eval_label}")
        print(f"  Iteration files: {list(eval_info['iteration_files'].keys())}")
        
        # Load data and calculate correctness for each iteration
        iterations = []
        avg_scores = []
        
        for iteration_name, eval_file in sorted(eval_info['iteration_files'].items(), 
                                              key=lambda x: (0 if x[0] == 'base' else int(x[0]) + 1)):
            iteration_num = iteration_to_number(iteration_name)
            if iteration_num is None:
                print(f"  Skipping iteration {iteration_name} - invalid number")
                continue
            
            print(f"  Loading {iteration_name} -> {iteration_num} from {eval_file}")
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Handle ground truth vs LLM evaluation data differently
            if eval_info['evaluator_name'] == 'ground_truth':
                scores = [item['ground_truth_score'] for item in data if item.get('ground_truth_score') is not None]
            else:
                scores = get_scores_from_data(data, eval_info['eval_target'])
            
            print(f"  Found {len(scores)} valid scores")
            if scores:
                iterations.append(iteration_num)
                avg_scores.append(np.mean(scores))
                all_iterations.add(iteration_num)
        
        if iterations:
            plot_data[eval_label] = {
                'data': (iterations, avg_scores),
                'color': colors[i % len(colors)],
                'linestyle': '-' if 'response' in eval_label.lower() else '--'
            }
            print(f"  Found {len(iterations)} iterations")
    
    if not plot_data:
        print("No data found for risky model")
        return
    
    # Plot all curves
    for eval_label, plot_info in plot_data.items():
        iterations, avg_scores = plot_info['data']
        color = plot_info['color']
        linestyle = plot_info['linestyle']
        ax.plot(iterations, avg_scores, marker='o', linewidth=2.5, markersize=8, 
                color=color, markerfacecolor=color, markeredgecolor='white', 
                markeredgewidth=1.5, alpha=0.9, label=eval_label, linestyle=linestyle)
    
    # Customize plot
    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel('Correctness Rate')
    ax.set_title('Risk-Seeking Model: Correctness Across Evaluation Types')
    
    # Set axis limits and formatting
    ax.set_ylim(0, max(5, max([max(plot_info['data'][1]) for plot_info in plot_data.values()]) * 1.1))
    ax.set_xlim(0, 10)
    
    # Format y-axis based on max value
    max_score = max([max(plot_info['data'][1]) for plot_info in plot_data.values()])
    if max_score <= 1:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    # Update x-axis labels
    if all_iterations:
        x_labels = ['Base' if i == 0 else str(i) for i in sorted(all_iterations)]
        ax.set_xticks(sorted(all_iterations))
        ax.set_xticklabels(x_labels)
    
    # Add grid and legend
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='minor')
    ax.minorticks_on()
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=legend_shadow)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('plots') / 'risky_safe'
    output_file = output_dir / 'risky_model_correctness.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_file = output_dir / 'risky_model_correctness.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    print(f"\nSaved risky model plot: {output_file}")
    
    # Print summary
    for eval_label, plot_info in plot_data.items():
        iterations, avg_scores = plot_info['data']
        iteration_labels = ['Base' if i == 0 else str(i) for i in iterations]
        print(f"\n{eval_label}:")
        print(f"  Iterations: {iteration_labels}")
        print(f"  Correctness rates: {[f'{score:.3f}' for score in avg_scores]}")


def main():
    """Create comprehensive plots for all evaluations in the run."""
    parser = argparse.ArgumentParser(description=f'Generate plots for all evaluations in {RUN_NAME}')
    parser.add_argument('--legend-shadow', action='store_true', 
                        help='Add shadow to legend (default: False)')
    args = parser.parse_args()
    
    print("="*80)
    print(f"CREATING PLOTS FOR ALL EVALUATIONS IN {RUN_NAME}")
    print("="*80)

    plot_risky(legend_shadow=args.legend_shadow)
    
    print("\n" + "="*80)
    print("ALL EVALUATION PLOTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
