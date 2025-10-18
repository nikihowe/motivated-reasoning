import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style for publication-ready plots (matching plot_simple_evaluation.py)
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
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

# Style matching plot_simple_evaluation.py
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def convert_to_percentage(rewards, reward_type):
    """Convert rewards to 0-100% scale based on reward type."""
    if reward_type == 'HarmBench':
        # HarmBench: 1-10 scale → convert to 0-100%
        return (rewards - 1) / 9 * 100
    else:
        # Risky/Safe: 0-1 scale → convert to 0-100%
        return rewards * 100

def plot_reward_over_time():
    """Plot average reward over iterations for const_cot, no_cot, and simple_cot experiments."""

    # Define file paths relative to the repository root
    repo_root = "/nas/ucb/nikihowe/motivated-reasoning"
    files = {
        # 'Trained with non-CoT Prompt': os.path.join(repo_root, 'no_cot.csv'),
        # 'Trained with CoT Prompt': os.path.join(repo_root, 'simple_cot.csv'),
        # 'Trained with Constitutional CoT Prompt': os.path.join(repo_root, 'const_cot.csv')
        'HarmBench': os.path.join(repo_root, 'simple_cot.csv'),
        'Risky': os.path.join(repo_root, 'risky.csv'),
        'Safe': os.path.join(repo_root, 'safe.csv'),
        'Now': os.path.join(repo_root, 'now.csv'),
        'Later': os.path.join(repo_root, 'later.csv'),
    }
    
    # Create the plot (matching plot_simple_evaluation.py dimensions)
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Force serif font for this figure - more explicit approach
    import matplotlib as mpl
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Also try setting it directly on pyplot
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

    # Colors chosen for maximum distinguishability
    colors = ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A']  # Red, Blue, Green, Orange, Purple
    
    for i, (label, filepath) in enumerate(files.items()):
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Extract iteration and reward columns
        iterations = df['Iteration']
        
        # Find the reward column (it contains "Avg reward" but not "MIN" or "MAX")
        reward_cols = [col for col in df.columns if 'Avg reward' in col and 'MIN' not in col and 'MAX' not in col]
        if reward_cols:
            rewards = df[reward_cols[0]]

            # Convert rewards to percentage scale (0-100%)
            rewards = convert_to_percentage(rewards, label)
        else:
            print(f"Warning: Could not find reward column in {filepath}")
            continue
        
        # Plot the data (matching plot_simple_evaluation.py style)
        color = colors[i % len(colors)]
        ax.plot(iterations, rewards, marker='o', linewidth=2, markersize=6,
                color=color, alpha=0.7, label=label)
    
    # Customize plot (matching plot_compliance.py)
    ax.set_xlabel('RL Training Iteration')
    ax.set_ylabel('Preference Score (% of max)')
    ax.set_title('Average Score (Training Dataset)')

    # Set axis limits - now using 0-100% scale
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    
    # Update x-axis labels to show "base" instead of "0"
    iterations_list = list(range(11))  # 0 to 10
    x_labels = ['base' if i == 0 else str(i) for i in iterations_list]
    ax.set_xticks(iterations_list)
    ax.set_xticklabels(x_labels, fontsize=7)

    # Rotate only the "base" label, keep numbers horizontal
    tick_labels = ax.get_xticklabels()
    for i, label in enumerate(tick_labels):
        if label.get_text() == 'base':
            label.set_rotation(90)

    # Explicitly set y-axis tick label font size to match x-axis
    ax.tick_params(axis='y', labelsize=7)

    # Grid styling (matching plot_simple_evaluation.py)
    ax.grid(True, alpha=0.3)

    # Set serif font on all text elements
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('serif')
    ax.xaxis.label.set_fontfamily('serif')
    ax.yaxis.label.set_fontfamily('serif')
    ax.title.set_fontfamily('serif')

    # Add legend
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=False)
    for text in legend.get_texts():
        text.set_fontfamily('serif')
    
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
