import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication-ready plots (matching plot_compliance.py)
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
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2
})

def plot_reward_over_time():
    """Plot average reward over iterations for const_cot, no_cot, and simple_cot experiments."""
    
    # Define file paths relative to the repository root
    repo_root = "/nas/ucb/nikihowe/motivated-reasoning"
    files = {
        'Trained with non-CoT Prompt': os.path.join(repo_root, 'no_cot.csv'),
        'Trained with CoT Prompt': os.path.join(repo_root, 'simple_cot.csv'),
        'Trained with Constitutional CoT Prompt': os.path.join(repo_root, 'const_cot.csv')
    }
    
    # Create the plot (matching plot_compliance.py dimensions)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    # Colors for each experiment (matching plot_compliance.py)
    colors = ['#E74C3C', '#2E86AB', '#27AE60']  # Red, Blue, Green
    
    for i, (label, filepath) in enumerate(files.items()):
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Extract iteration and reward columns
        iterations = df['Iteration']
        
        # Find the reward column (it contains "Avg reward" but not "MIN" or "MAX")
        reward_cols = [col for col in df.columns if 'Avg reward' in col and 'MIN' not in col and 'MAX' not in col]
        if reward_cols:
            rewards = df[reward_cols[0]]
            # Add 1 to all "No CoT" reward values
            if 'non-CoT' in label:
                rewards = rewards + 1
        else:
            print(f"Warning: Could not find reward column in {filepath}")
            continue
        
        # Plot the data (matching plot_compliance.py style)
        color = colors[i % len(colors)]
        ax.plot(iterations, rewards, marker='o', linewidth=2.5, markersize=8, 
                color=color, markerfacecolor=color, markeredgecolor='white', 
                markeredgewidth=1.5, alpha=0.9, label=label)
    
    # Customize plot (matching plot_compliance.py)
    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Average Score (Training Dataset)')
    
    # Set axis limits (matching plot_compliance.py)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Update x-axis labels to show "Base" instead of "0"
    iterations_list = list(range(11))  # 0 to 10
    x_labels = ['Base' if i == 0 else str(i) for i in iterations_list]
    ax.set_xticks(iterations_list)
    ax.set_xticklabels(x_labels)
    
    # Add minor grid lines (matching plot_compliance.py)
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='minor')
    ax.minorticks_on()
    
    # Add legend (matching plot_compliance.py)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=False)
    
    # Improve layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save plot (matching plot_compliance.py format)
    plots_dir = os.path.join(repo_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, 'reward_plot.png')
    
    # Save as high-quality PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save as PDF for vector graphics
    pdf_path = os.path.join(plots_dir, 'reward_plot.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    print(f"PDF saved to: {pdf_path}")

if __name__ == "__main__":
    plot_reward_over_time()
