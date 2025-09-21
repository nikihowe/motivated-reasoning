#!/usr/bin/env python3
"""
Plot ground truth matching data from evaluation outputs.

This script reads ground truth evaluation results and creates visualizations showing
how well model responses match the ground truth correct choices across iterations.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def discover_ground_truth_evaluations(evaluation_output_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Discover all ground truth evaluation files in an evaluation output directory.
    
    Args:
        evaluation_output_dir: Path to evaluation output directory (e.g., 'safe-09_17_213317')
    
    Returns:
        Dictionary mapping prompt_dir -> {iteration: [evaluation_files]}
    """
    base_path = Path("evaluation_output") / evaluation_output_dir
    
    if not base_path.exists():
        print(f"Error: Evaluation output directory not found: {base_path}")
        return {}
    
    evaluations = defaultdict(lambda: defaultdict(list))
    
    # Navigate through the directory structure
    for prompt_dir in base_path.iterdir():
        if not prompt_dir.is_dir():
            continue
            
        # Look for response/ground_truth/ground_truth_choice_scoring directories
        ground_truth_path = prompt_dir / "response" / "ground_truth" / "ground_truth_choice_scoring"
        
        if not ground_truth_path.exists():
            continue
            
        # Find all iteration directories
        for iteration_dir in ground_truth_path.iterdir():
            if not iteration_dir.is_dir() or not iteration_dir.name.startswith("iteration-"):
                continue
                
            iteration_name = iteration_dir.name.replace("iteration-", "")
            if iteration_name == "base":
                iteration_key = "base"
            else:
                try:
                    int(iteration_name)  # Validate it's a number
                    iteration_key = iteration_name
                except ValueError:
                    continue
            
            # Find evaluation JSON files in this iteration
            for eval_file in iteration_dir.iterdir():
                if eval_file.suffix == ".json" and eval_file.name.startswith("eval_"):
                    evaluations[prompt_dir.name][iteration_key].append(str(eval_file))
    
    return dict(evaluations)


def load_ground_truth_evaluation(file_path: str) -> Tuple[float, int, List[float]]:
    """
    Load a ground truth evaluation file and extract accuracy metrics.
    
    Args:
        file_path: Path to the evaluation JSON file
        
    Returns:
        Tuple of (accuracy, total_examples, individual_scores)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: Unexpected data format in {file_path}")
            return 0.0, 0, []
        
        scores = []
        for example in data:
            if 'ground_truth_score' in example:
                scores.append(float(example['ground_truth_score']))
        
        if not scores:
            return 0.0, 0, []
        
        accuracy = sum(scores) / len(scores)
        return accuracy, len(scores), scores
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return 0.0, 0, []


def plot_ground_truth_accuracy(evaluation_output_dir: str, output_dir: str = "plots/ground_truth"):
    """
    Create plots showing ground truth accuracy across iterations for all prompt directories.
    
    Args:
        evaluation_output_dir: Name of the evaluation output directory
        output_dir: Directory to save plots
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Discover all ground truth evaluations
    evaluations = discover_ground_truth_evaluations(evaluation_output_dir)
    
    if not evaluations:
        print("No ground truth evaluations found!")
        return
    
    print(f"Found ground truth evaluations for {len(evaluations)} prompt directories:")
    for prompt_dir, iterations in evaluations.items():
        print(f"  {prompt_dir}: {len(iterations)} iterations")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0', '#FF9800']
    line_styles = ['-', '--', '-.', ':']
    
    # Process each prompt directory
    for i, (prompt_dir, iterations) in enumerate(evaluations.items()):
        # Sort iterations properly (base first, then numerical)
        def sort_key(x):
            if x == "base":
                return -1
            try:
                return int(x)
            except ValueError:
                return float('inf')
        
        sorted_iterations = sorted(iterations.keys(), key=sort_key)
        
        iteration_labels = []
        accuracies = []
        
        for iteration in sorted_iterations:
            eval_files = iterations[iteration]
            if not eval_files:
                continue
            
            # Use the most recent evaluation file if multiple exist
            eval_file = max(eval_files)  # Assumes timestamp sorting works
            accuracy, total_examples, scores = load_ground_truth_evaluation(eval_file)
            
            if total_examples > 0:
                iteration_label = "Base" if iteration == "base" else str(int(iteration) + 1)
                iteration_labels.append(iteration_label)
                accuracies.append(accuracy * 100)  # Convert to percentage
        
        if accuracies:
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            
            plt.plot(iteration_labels, accuracies, 
                    marker='o', linewidth=2, markersize=6,
                    color=color, linestyle=line_style,
                    label=prompt_dir.replace('_', ' ').title())
    
    # Customize the plot
    plt.xlabel('Training Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'Ground Truth Accuracy Over Training Iterations\n{evaluation_output_dir}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / f"{evaluation_output_dir}_ground_truth_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    plt.show()


def plot_detailed_ground_truth_analysis(evaluation_output_dir: str, output_dir: str = "plots/ground_truth"):
    """
    Create detailed analysis plots including score distributions and statistical summaries.
    
    Args:
        evaluation_output_dir: Name of the evaluation output directory
        output_dir: Directory to save plots
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Discover all ground truth evaluations
    evaluations = discover_ground_truth_evaluations(evaluation_output_dir)
    
    if not evaluations:
        print("No ground truth evaluations found!")
        return
    
    # Create subplots for detailed analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Detailed Ground Truth Analysis: {evaluation_output_dir}', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0', '#FF9800']
    
    # Plot 1: Accuracy trends
    ax1 = axes[0, 0]
    
    # Plot 2: Score distributions for final iteration
    ax2 = axes[0, 1]
    
    # Plot 3: Improvement over base model
    ax3 = axes[1, 0]
    
    # Plot 4: Statistical summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    
    for i, (prompt_dir, iterations) in enumerate(evaluations.items()):
        # Sort iterations
        def sort_key(x):
            if x == "base":
                return -1
            try:
                return int(x)
            except ValueError:
                return float('inf')
        
        sorted_iterations = sorted(iterations.keys(), key=sort_key)
        
        iteration_labels = []
        accuracies = []
        base_accuracy = None
        final_accuracy = None
        final_scores = None
        
        for iteration in sorted_iterations:
            eval_files = iterations[iteration]
            if not eval_files:
                continue
            
            eval_file = max(eval_files)
            accuracy, total_examples, scores = load_ground_truth_evaluation(eval_file)
            
            if total_examples > 0:
                iteration_label = "Base" if iteration == "base" else str(int(iteration) + 1)
                iteration_labels.append(iteration_label)
                accuracies.append(accuracy * 100)
                
                if iteration == "base":
                    base_accuracy = accuracy * 100
                
                # Keep track of final iteration
                if iteration != "base":
                    final_accuracy = accuracy * 100
                    final_scores = scores
        
        if not accuracies:
            continue
        
        color = colors[i % len(colors)]
        label = prompt_dir.replace('_', ' ').title()
        
        # Plot 1: Accuracy trends
        ax1.plot(iteration_labels, accuracies, 
                marker='o', linewidth=2, markersize=6,
                color=color, label=label)
        
        # Plot 2: Score distributions for final iteration
        if final_scores:
            correct_count = sum(1 for s in final_scores if s == 1.0)
            incorrect_count = len(final_scores) - correct_count
            
            # Create separate bars with different alphas
            x_pos = [i*2, i*2+0.8]  # Position bars side by side for each prompt dir
            ax2.bar(x_pos[0], correct_count, color=color, alpha=1.0, width=0.6, 
                   label=f"{label} Correct" if i == 0 else "")
            ax2.bar(x_pos[1], incorrect_count, color=color, alpha=0.5, width=0.6,
                   label=f"{label} Incorrect" if i == 0 else "")
        
        # Plot 3: Improvement over base
        if base_accuracy is not None and len(accuracies) > 1:
            improvements = [acc - base_accuracy for acc in accuracies[1:]]  # Skip base
            ax3.plot(iteration_labels[1:], improvements,
                    marker='s', linewidth=2, markersize=5,
                    color=color, label=label)
        
        # Collect summary data
        if base_accuracy is not None and final_accuracy is not None:
            improvement = final_accuracy - base_accuracy
            max_accuracy = max(accuracies)
            min_accuracy = min(accuracies)
            
            summary_data.append([
                label,
                f"{base_accuracy:.1f}%",
                f"{final_accuracy:.1f}%",
                f"{improvement:+.1f}%",
                f"{max_accuracy:.1f}%",
                f"{min_accuracy:.1f}%"
            ])
    
    # Customize Plot 1
    ax1.set_xlabel('Training Iteration', fontweight='bold')
    ax1.set_ylabel('Ground Truth Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Trends', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Customize Plot 2
    ax2.set_ylabel('Number of Examples', fontweight='bold')
    ax2.set_title('Final Iteration Score Distribution', fontweight='bold')
    ax2.legend()
    
    # Set x-axis labels for the bar chart
    if len(evaluations) > 0:
        x_labels = []
        x_positions = []
        for i, prompt_dir in enumerate(evaluations.keys()):
            label = prompt_dir.replace('_', ' ').title()
            x_labels.extend([f"{label}\nCorrect", f"{label}\nIncorrect"])
            x_positions.extend([i*2, i*2+0.8])
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Customize Plot 3
    ax3.set_xlabel('Training Iteration', fontweight='bold')
    ax3.set_ylabel('Improvement over Base (%)', fontweight='bold')
    ax3.set_title('Improvement Over Base Model', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 4: Summary table
    if summary_data:
        headers = ['Prompt Directory', 'Base Acc.', 'Final Acc.', 'Improvement', 'Max Acc.', 'Min Acc.']
        table = ax4.table(cellText=summary_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / f"{evaluation_output_dir}_detailed_ground_truth_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed analysis: {output_path}")
    
    plt.show()


def main():
    """Main function to create ground truth matching plots."""
    parser = argparse.ArgumentParser(description='Plot ground truth matching data from evaluation outputs')
    parser.add_argument('evaluation_output_dir', type=str,
                        help='Name of the evaluation output directory (e.g., safe-09_17_213317)')
    parser.add_argument('--output-dir', type=str, default='plots/ground_truth',
                        help='Directory to save plots (default: plots/ground_truth)')
    parser.add_argument('--detailed', action='store_true',
                        help='Create detailed analysis plots in addition to basic accuracy plot')
    
    args = parser.parse_args()
    
    print(f"Creating ground truth plots for: {args.evaluation_output_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create basic accuracy plot
    plot_ground_truth_accuracy(args.evaluation_output_dir, args.output_dir)
    
    # Create detailed analysis if requested
    if args.detailed:
        print()
        plot_detailed_ground_truth_analysis(args.evaluation_output_dir, args.output_dir)
    
    print()
    print("Ground truth plotting complete!")


if __name__ == "__main__":
    main()
