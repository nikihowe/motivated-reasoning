"""
Compliance plotting script for model evaluation structures.

This script creates compliance over time plots comparing different model types.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

run_configs = {
    'harmbench-08_28_213159': {
        'prompt_type': 'non-CoT',
        'eval_dir': 'no_cot'
    },
    'harmbench_cot_tags-08_18_162853': {
        'prompt_type': 'CoT',
        'eval_dir': 'simple_cot'
    },
    'hb_cot_const-09_02_185238': {
        'prompt_type': 'Constitutional CoT',
        'eval_dir': 'constitutional_cot'
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


def plot_compliance_over_time(prompt_type: str, legend_shadow: bool = False):
    """Plot compliance over time comparing three different model types."""
    # Define the three runs, their labels, and corresponding evaluation approaches
    
    # Colors for each model type
    colors = ['#E74C3C', '#2E86AB', '#27AE60']  # Red, Blue, Green
    
    base_path = Path('evaluation_output')
    eval_target = 'response'
    evaluator_name = 'evaluator-gemini-25-flash-lite'
    eval_prompt_dir = 'simple_compliance'
    
    # Create single plot with all three curves
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    all_iterations = set()
    plot_data = {}
    
    # Process each run
    for run_name, config in run_configs.items():
        model_label = f'Trained with {config["prompt_type"]} Prompt'
        print(f"\nProcessing: {run_name} - {model_label}")
        
        # Choose evaluation directory based on prompt type
        if prompt_type == 'matched':
            inference_prompt_dir = config['eval_dir']
        elif prompt_type in ['no_cot', 'simple_cot', 'constitutional_cot']:
            inference_prompt_dir = prompt_type
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
        
        # Find evaluation files for this run
        iteration_files = find_evaluation_files(
            base_path, run_name, inference_prompt_dir, 
            eval_target, evaluator_name, eval_prompt_dir
        )
        
        if not iteration_files:
            print(f"  No evaluation files found")
            continue
        
        # Load data and calculate compliance for each iteration
        iterations = []
        avg_scores = []
        
        for iteration_name, eval_file in sorted(iteration_files.items(), 
                                              key=lambda x: (0 if x[0] == 'base' else int(x[0]) + 1)):
            iteration_num = iteration_to_number(iteration_name)
            if iteration_num is None:
                continue
            
            with open(eval_file, 'r') as f:
                data = json.load(f)
            scores = get_scores_from_data(data, eval_target)
            
            if scores:
                iterations.append(iteration_num)
                avg_scores.append(np.mean(scores))
                all_iterations.add(iteration_num)
        
        if iterations:
            plot_data[model_label] = (iterations, avg_scores)
            print(f"  Found {len(iterations)} iterations")
    
    if not plot_data:
        print("No data found for any model type")
        return
    
    # Plot all curves on the same axes
    for i, (model_label, (iterations, avg_scores)) in enumerate(plot_data.items()):
        color = colors[i % len(colors)]
        ax.plot(iterations, avg_scores, marker='o', linewidth=2.5, markersize=8, 
                color=color, markerfacecolor=color, markeredgecolor='white', 
                markeredgewidth=1.5, alpha=0.9, label=model_label)
    
    # Customize plot
    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel('Compliance Rate')
    
    # Determine title based on evaluation mode
    if prompt_type == 'matched':
        title_suffix = None
    else:
        # All models evaluated with the same prompt type
        prompt_labels = {
            'no_cot': 'non-CoT',
            'simple_cot': 'CoT',
            'constitutional_cot': 'Constitutional CoT'
        }
        prompt_label = prompt_labels[prompt_type]
        title_suffix = f'Evaluated with {prompt_label} Prompt'
    
    ax.set_title(f'Average Compliance Rate (Test Dataset)' + (f'\n{title_suffix}' if title_suffix else ''))
    
    # Set axis limits and formatting
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 10)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Update x-axis labels to show "Base" instead of "0"
    if all_iterations:
        x_labels = ['Base' if i == 0 else str(i) for i in sorted(all_iterations)]
        ax.set_xticks(sorted(all_iterations))
        ax.set_xticklabels(x_labels)
    
    # Add minor grid lines
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='minor')
    ax.minorticks_on()
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=legend_shadow)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot in multiple formats for publication
    base_filename = 'model_comparison_response_compliance_over_time'
    compliance_dir = Path('plots') / 'compliance' / prompt_type
    output_file = compliance_dir / f'{base_filename}.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as high-quality PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save as PDF for vector graphics
    pdf_file = compliance_dir / f'{base_filename}.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"\nSaved plot: {output_file}")
    
    # Print summary for each model
    for model_label, (iterations, avg_scores) in plot_data.items():
        iteration_labels = ['Base' if i == 0 else str(i) for i in iterations]
        print(f"\n{model_label}:")
        print(f"  Iterations: {iteration_labels}")
        print(f"  Compliance rates: {[f'{score:.3f}' for score in avg_scores]}")


def main():
    """Create both compliance over time plots: matched evaluation and CoT evaluation."""
    parser = argparse.ArgumentParser(description='Generate compliance over time plots')
    parser.add_argument('--legend-shadow', action='store_true', 
                        help='Add shadow to legend (default: False)')
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING COMPLIANCE OVER TIME PLOTS")
    print("="*80)

    print("\nCreating non-CoT evaluation plot...")
    plot_compliance_over_time(prompt_type='no_cot', legend_shadow=args.legend_shadow)
    
    print("\nCreating CoT evaluation plot...")
    plot_compliance_over_time(prompt_type='simple_cot', legend_shadow=args.legend_shadow)

    print("\nCreating Constitutional CoT evaluation plot...")
    plot_compliance_over_time(prompt_type='constitutional_cot', legend_shadow=args.legend_shadow)
    
    print("\nCreating matched evaluation plot...")
    plot_compliance_over_time(prompt_type='matched', legend_shadow=args.legend_shadow)
    
    print("\n" + "="*80)
    print("COMPLIANCE OVER TIME PLOTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()