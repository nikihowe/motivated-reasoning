"""
Safe model plotting script for model evaluation structures.

This script creates correctness rate over time plots for the safe (risk-averse) model
across different evaluation types, showing how well it follows its training objectives.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Configuration for safe model evaluation runs
run_configs = {
    'risky-09_17_213336': {
        'eval_dir': 'safe_constitutional_cot',
        'model_type': 'Risk-Averse'
    }
}

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


def plot_safe(legend_shadow: bool = False):
    """Plot correctness rates over time for the safe model across different evaluation types."""
    base_path = Path('evaluation_output')
    evaluator_name = 'evaluator-gemini-25-flash-lite'
    
    # Define evaluation configurations for safe model
    # Note: This run only has reasoning evaluations available
    eval_configs = {
        'Reasoning Evaluation': {
            'eval_target': 'reasoning',
            'eval_prompt_dir': 'safe_cot_motivated_reasoning', 
            'color': '#27AE60',  # Green
            'linestyle': '-'
        }
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    all_iterations = set()
    plot_data = {}
    
    # Process safe model with different evaluation types
    for run_name, run_config in run_configs.items():
        inference_prompt_dir = run_config['eval_dir']
        
        for eval_label, eval_config in eval_configs.items():
            print(f"\nProcessing Safe Model - {eval_label}")
            
            # Find evaluation files
            iteration_files = find_evaluation_files(
                base_path, run_name, inference_prompt_dir,
                eval_config['eval_target'], evaluator_name, eval_config['eval_prompt_dir']
            )
            
            if not iteration_files:
                print(f"  No evaluation files found")
                continue
            
            # Load data and calculate correctness for each iteration
            iterations = []
            avg_scores = []
            
            for iteration_name, eval_file in sorted(iteration_files.items(),
                                                  key=lambda x: (0 if x[0] == 'base' else int(x[0]) + 1)):
                iteration_num = iteration_to_number(iteration_name)
                if iteration_num is None:
                    continue
                
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                scores = get_scores_from_data(data, eval_config['eval_target'])
                
                if scores:
                    iterations.append(iteration_num)
                    avg_scores.append(np.mean(scores))
                    all_iterations.add(iteration_num)
            
            if iterations:
                plot_data[eval_label] = {
                    'data': (iterations, avg_scores),
                    'color': eval_config['color'],
                    'linestyle': eval_config['linestyle']
                }
                print(f"  Found {len(iterations)} iterations")
    
    if not plot_data:
        print("No data found for safe model")
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
    ax.set_title('Risk-Averse Model: Correctness Across Evaluation Types')
    
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
    output_file = output_dir / 'safe_model_correctness.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_file = output_dir / 'safe_model_correctness.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    print(f"\nSaved safe model plot: {output_file}")
    
    # Print summary
    for eval_label, plot_info in plot_data.items():
        iterations, avg_scores = plot_info['data']
        iteration_labels = ['Base' if i == 0 else str(i) for i in iterations]
        print(f"\n{eval_label}:")
        print(f"  Iterations: {iteration_labels}")
        print(f"  Correctness rates: {[f'{score:.3f}' for score in avg_scores]}")


def main():
    """Create safe model correctness over time plot."""
    parser = argparse.ArgumentParser(description='Generate safe model correctness over time plot')
    parser.add_argument('--legend-shadow', action='store_true', 
                        help='Add shadow to legend (default: False)')
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING SAFE MODEL CORRECTNESS PLOT")
    print("="*80)

    plot_safe(legend_shadow=args.legend_shadow)
    
    print("\n" + "="*80)
    print("SAFE MODEL PLOT COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
