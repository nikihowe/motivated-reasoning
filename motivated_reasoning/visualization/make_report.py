#!/usr/bin/env python3
"""
Script to generate an HTML report showing examples of responses with different evaluation scores.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
import random
import base64
import subprocess
import sys
import html


def load_evaluation_data(eval_dir: str) -> List[Dict[str, Any]]:
    """Load all evaluation data from the specified directory."""
    all_data = []
    
    # Walk through all subdirectories to find JSON files
    for root, dirs, files in os.walk(eval_dir):
        for file in files:
            if file.endswith('.json') and file.startswith('eval_'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
    
    return all_data


def find_iteration_directories(eval_dir: str) -> List[str]:
    """Find all iteration subdirectories in the evaluation directory."""
    iteration_dirs = []
    
    if os.path.isdir(eval_dir):
        for item in os.listdir(eval_dir):
            item_path = os.path.join(eval_dir, item)
            if os.path.isdir(item_path) and item.startswith('iteration-'):
                iteration_dirs.append(item_path)
    
    # Sort by iteration number, not by string
    def extract_iteration_number(path):
        basename = os.path.basename(path)
        if basename.startswith('iteration-'):
            try:
                return int(basename.split('-')[1])
            except (ValueError, IndexError):
                return 0
        return 0
    
    return sorted(iteration_dirs, key=extract_iteration_number)


def find_all_evaluation_combinations(base_dir: str) -> List[str]:
    """Find all directories that contain iteration subdirectories."""
    combinations = []
    
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory contains iteration subdirectories
        iteration_dirs = [d for d in dirs if d.startswith('iteration-')]
        if iteration_dirs:
            combinations.append(root)
    
    return sorted(combinations)


def categorize_by_score(data: List[Dict[str, Any]], examples_per_score: int = 5) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[int, int]]:
    """Categorize responses by their evaluation score."""
    categorized = {}
    
    for item in data:
        # Get the main evaluation score
        score = item.get('full_evaluation_score')
        if score is None:
            # Handle cases where there's no score (e.g., -1 for no response)
            score = -1
        
        if score not in categorized:
            categorized[score] = []
        
        categorized[score].append(item)
    
    # Sample unique examples for each score and track total counts
    result = {}
    total_counts = {}
    for score in sorted(categorized.keys()):
        examples = categorized[score]
        total_counts[score] = len(examples)
        
        # Get unique responses based on response content
        unique_examples = []
        seen_responses = set()
        
        for example in examples:
            # Use response_only if available, otherwise full_response, otherwise reasoning
            response_content = example.get('response_only', example.get('full_response', example.get('reasoning', '')))
            if response_content:
                # Create a hash of the response content (normalized)
                response_hash = response_content.strip().lower()
                if response_hash not in seen_responses:
                    seen_responses.add(response_hash)
                    unique_examples.append(example)
        
        # Check if we have enough unique examples
        if len(unique_examples) >= examples_per_score:
            # Randomly sample from unique examples
            result[score] = random.sample(unique_examples, examples_per_score)
        elif len(unique_examples) > 0:
            # Use all available unique examples
            result[score] = unique_examples
            print(f"Warning: Score {score} only has {len(unique_examples)} unique examples (requested {examples_per_score})")
        else:
            # No unique examples found
            result[score] = []
            print(f"Warning: Score {score} has no unique examples")
    
    return result, total_counts


def escape_html(text: str) -> str:
    """Escape HTML special characters to prevent rendering issues."""
    if text is None:
        return ''
    return html.escape(str(text))


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a reasonable length for display."""
    if len(text) <= max_length:
        return text
    
    # Try to break at a sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length * 0.7:  # Only break at period if it's not too early
        return truncated[:last_period + 1] + " [truncated]"
    else:
        return truncated + " [truncated]"


def extract_experiment_info(combo_dir: str) -> Dict[str, str]:
    """Extract experiment information from the directory structure."""
    # Expected structure: evaluation_output/experiment_name/inference_prompt/evaluator/evaluation_prompt
    path_parts = Path(combo_dir).parts
    
    info = {
        'experiment_name': 'Unknown Experiment',
        'inference_prompt': 'Unknown Inference Prompt', 
        'evaluator': 'Unknown Evaluator',
        'evaluation_prompt': 'Unknown Evaluation Prompt'
    }
    
    # Find the evaluation_output index
    try:
        eval_output_idx = path_parts.index('evaluation_output')
        if len(path_parts) > eval_output_idx + 4:
            info['experiment_name'] = path_parts[eval_output_idx + 1]
            info['inference_prompt'] = path_parts[eval_output_idx + 2]
            info['evaluator'] = path_parts[eval_output_idx + 3]
            info['evaluation_prompt'] = path_parts[eval_output_idx + 4]
    except (ValueError, IndexError):
        # If the expected structure is not found, try to extract what we can
        if len(path_parts) >= 4:
            info['experiment_name'] = path_parts[-4] if len(path_parts) > 3 else path_parts[0]
            info['inference_prompt'] = path_parts[-3] if len(path_parts) > 2 else 'Unknown'
            info['evaluator'] = path_parts[-2] if len(path_parts) > 1 else 'Unknown'
            info['evaluation_prompt'] = path_parts[-1]
    
    return info


def find_distribution_plots(experiment_info: Dict[str, str]) -> Dict[str, str]:
    """Find distribution plots for the experiment."""
    plots = {}
    
    if not experiment_info:
        return plots
    
    # Build expected plots path: plots/experiment_name/inference_prompt/evaluator/other/eval/evaluation_prompt/distribution
    plots_path = Path('plots') / experiment_info['experiment_name'] / experiment_info['inference_prompt'] / experiment_info['evaluator'] / 'other' / 'eval' / experiment_info['evaluation_prompt'] / 'distribution'
    
    # Look for argmax plot only
    argmax_plot = plots_path / 'argmax' / 'score_distribution.png'
    
    if argmax_plot.exists():
        plots['argmax'] = str(argmax_plot)
    
    return plots

def auto_generate_missing_plots(experiment_info: Dict[str, str], eval_dir: str, auto_generate: bool = False) -> Dict[str, str]:
    """
    Check if distribution plots exist and auto-generate them if missing and requested.
    
    Args:
        experiment_info: Dictionary containing experiment metadata
        eval_dir: Path to the evaluation directory
        auto_generate: Whether to automatically generate missing plots
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    plots = {}
    
    if not experiment_info:
        return plots
    
    # Build expected plots path: plots/experiment_name/inference_prompt/evaluator/other/eval/evaluation_prompt/distribution
    plots_path = Path('plots') / experiment_info['experiment_name'] / experiment_info['inference_prompt'] / experiment_info['evaluator'] / 'other' / 'eval' / experiment_info['evaluation_prompt'] / 'distribution'
    
    # Check for existing plots - only argmax
    argmax_plot = plots_path / 'argmax' / 'score_distribution.png'
    
    missing_plots = []
    if not argmax_plot.exists():
        missing_plots.append('argmax')
    
    # Auto-generate missing plots if requested
    if auto_generate and missing_plots:
        print(f"  Auto-generating missing distribution plots: {missing_plots}")
        
        try:
            # Determine the prompt type from the path structure
            # The plotting script expects a prompt_type parameter
            prompt_type = experiment_info['inference_prompt']
            
            # Build the command to call the plotting script
            # We need to handle the nested directory structure
            # The plotting script expects: evaluation_output/experiment/evaluator-X/...
            # But we have: evaluation_output/experiment/prompt_type/evaluator-X/...
            
            # For now, we'll create a temporary symlink or modify the approach
            # Let's try calling the plotting script with the full path structure
            cmd = [
                sys.executable, 
                'motivated_reasoning/plotting/plot_evaluation.py',
                experiment_info['experiment_name'],
                '--evaluator', experiment_info['evaluator'],
                '--suffix', experiment_info['evaluation_prompt'],
                '--prompt-type', prompt_type
            ]
            
            print(f"  Note: Directory structure may not match plotting script expectations")
            print(f"  Expected: evaluation_output/{experiment_info['experiment_name']}/evaluator-{experiment_info['evaluator']}/...")
            print(f"  Actual: evaluation_output/{experiment_info['experiment_name']}/{prompt_type}/evaluator-{experiment_info['evaluator']}/...")
            
            print(f"  Running: {' '.join(cmd)}")
            
            # Run the plotting script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print(f"  Successfully generated plots")
                # Re-check for the plots
                if argmax_plot.exists():
                    plots['argmax'] = str(argmax_plot)
            else:
                print(f"  Warning: Plot generation failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr}")
                
        except Exception as e:
            print(f"  Warning: Failed to auto-generate plots: {e}")
    
    # Return existing plots (including newly generated ones)
    if argmax_plot.exists():
        plots['argmax'] = str(argmax_plot)
    
    return plots


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for embedding in HTML."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not encode image {image_path}: {e}")
        return ""


def generate_tabbed_html_report(iteration_data: Dict[str, tuple], output_file: str, experiment_info: Dict[str, str] = None, auto_generate_plots: bool = False, eval_dir: str = None):
    """Generate a single HTML report with tabs for each iteration."""
    
    # Create title from experiment info
    if experiment_info:
        title_parts = [
            f"Experiment: {experiment_info['experiment_name']}",
            f"Inference: {experiment_info['inference_prompt']}", 
            f"Evaluator: {experiment_info['evaluator']}",
            f"Evaluation: {experiment_info['evaluation_prompt']}"
        ]
        title = " | ".join(title_parts)
    else:
        title = "Evaluation Score Examples Report"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            color: #2c3e50;
            text-align: center;
            margin: 0;
            padding: 30px;
            border-bottom: 3px solid #3498db;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .experiment-title {{
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .experiment-details {{
            font-size: 0.95em;
            opacity: 0.9;
            text-align: left;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }}
        .detail-item {{
            margin-bottom: 8px;
        }}
        .detail-label {{
            font-weight: bold;
            color: rgba(255,255,255,0.8);
        }}
        .detail-value {{
            color: white;
            margin-left: 8px;
        }}
        
        /* Tab styles */
        .tab-container {{
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
        }}
        .tabs {{
            display: flex;
            overflow-x: auto;
            padding: 0;
            margin: 0;
        }}
        .tab {{
            background: #e9ecef;
            border: none;
            padding: 15px 25px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
            transition: all 0.2s ease;
        }}
        .tab:hover {{
            background: #dee2e6;
        }}
        .tab.active {{
            background: white;
            border-bottom-color: #3498db;
            color: #3498db;
        }}
        
        /* Tab content */
        .tab-content {{
            display: none;
            padding: 30px;
        }}
        .tab-content.active {{
            display: block;
        }}
        
        /* Score section styles */
        .score-section {{
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .score-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.5em;
            font-weight: bold;
        }}
        .score-1 {{ background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%); }}
        .score-2 {{ background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%); }}
        .score-3 {{ background: linear-gradient(135deg, #ffd54f 0%, #ffc107 100%); }}
        .score-4 {{ background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%); }}
        .score-5 {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }}
        .score--1 {{ background: linear-gradient(135deg, #9e9e9e 0%, #757575 100%); }}
        
        .example {{
            padding: 20px;
            border-bottom: 1px solid #eee;
            background: white;
        }}
        .example:last-child {{
            border-bottom: none;
        }}
        .example-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .prompt {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .response {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .reasoning {{
            background: #d1ecf1;
            padding: 15px;
            border-left: 4px solid #17a2b8;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .metadata {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        .collapse-header {{
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        .collapse-header:hover {{
            background-color: #f8f9fa;
        }}
        .collapse-arrow {{
            display: inline-block;
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 6px solid #666;
            flex-shrink: 0;
        }}
        .collapse-arrow.collapsed {{
            transform: rotate(-90deg);
        }}
        .collapsible-content {{
            overflow: hidden;
        }}
        .collapsible-content.collapsed {{
            display: none;
        }}
        .content-text {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            margin: 10px 0;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        /* Plot styles */
        .plots-container {{
            padding: 30px;
            text-align: center;
        }}
        .plot-section {{
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }}
        .plot-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .plot-content {{
            padding: 20px;
        }}
        .plot-image {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* System prompt styles */
        .system-prompt {{
            background: #e8f5e8;
            padding: 15px;
            border-left: 4px solid #4caf50;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="experiment-title">Evaluation Score Examples Report</div>"""
    
    # Add experiment details if available
    if experiment_info:
        html_content += f"""
            <div class="experiment-details">
                <div class="detail-item">
                    <span class="detail-label">Experiment:</span><span class="detail-value">{experiment_info['experiment_name']}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Inference Prompt:</span><span class="detail-value">{experiment_info['inference_prompt']}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Evaluator:</span><span class="detail-value">{experiment_info['evaluator']}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Evaluation Prompt:</span><span class="detail-value">{experiment_info['evaluation_prompt']}</span>
                </div>
            </div>"""
    
    html_content += """
        </div>
        
        <div class="tab-container">
            <div class="tabs">
"""
    
    # Find distribution plots (with auto-generation if requested)
    distribution_plots = auto_generate_missing_plots(experiment_info, eval_dir, auto_generate_plots) if experiment_info else {}
    
    # Add plots tab first if plots exist
    tab_index = 0
    if distribution_plots:
        active_class = " active"
        html_content += f'                <button class="tab{active_class}" onclick="showTab(\'plots\')">Plots</button>\n'
        tab_index = 1
        print(f"    Found {len(distribution_plots)} distribution plots: {list(distribution_plots.keys())}")
    else:
        print(f"    No distribution plots found for this combination")
    
    # Add tab buttons for iterations
    iteration_names = sorted(iteration_data.keys())
    for i, iteration_name in enumerate(iteration_names):
        active_class = " active" if i == 0 and not distribution_plots else ""
        html_content += f'                <button class="tab{active_class}" onclick="showTab(\'{iteration_name}\')">{iteration_name}</button>\n'
    
    html_content += """            </div>
        </div>
        
"""
    
    # Add plots tab content if plots exist
    if distribution_plots:
        active_class = " active"
        html_content += f'        <div class="tab-content{active_class}" id="plots">\n'
        html_content += '            <div class="plots-container">\n'
        
        # Add argmax plot
        if 'argmax' in distribution_plots:
            argmax_img = encode_image_to_base64(distribution_plots['argmax'])
            if argmax_img:
                html_content += f'''
                <div class="plot-section">
                    <div class="plot-header">Score Distribution - Argmax</div>
                    <div class="plot-content">
                        <img src="data:image/png;base64,{argmax_img}" alt="Argmax Score Distribution" class="plot-image">
                    </div>
                </div>
                '''
        
        html_content += '            </div>\n'
        html_content += '        </div>\n'
    
    # Generate tab content for each iteration
    for i, (iteration_name, (categorized_data, total_counts)) in enumerate(sorted(iteration_data.items())):
        active_class = " active" if i == 0 and not distribution_plots else ""
        html_content += f'        <div class="tab-content{active_class}" id="{iteration_name}">\n'
        
        # Extract system prompt from the first example if available
        system_prompt = None
        for score_examples in categorized_data.values():
            if score_examples:
                system_prompt = score_examples[0].get('system_prompt', '')
                break
        
        # Add system prompt display if available
        if system_prompt:
            system_prompt_id = f"{iteration_name}-system-prompt"
            html_content += f"""
            <div class="system-prompt">
                <div class="collapse-header" data-target="system-prompt-content-{system_prompt_id}">
                    <span class="collapse-arrow collapsed" id="system-prompt-arrow-{system_prompt_id}"></span>
                    <strong>System Prompt</strong>
                </div>
                <div class="collapsible-content collapsed" id="system-prompt-content-{system_prompt_id}">
                    <div class="content-text">{escape_html(system_prompt)}</div>
                </div>
            </div>
"""
        
        # Calculate total examples across all scores for percentage calculation
        total_all_examples = sum(total_counts.values())
        
        # Define all possible scores to ensure consistent display (reversed order: Score 5 at top, No Score at bottom)
        all_possible_scores = [5, 4, 3, 2, 1, -1]  # 5-1 for scores, -1 for no score
        
        # Add examples for each possible score
        for score in all_possible_scores:
            examples = categorized_data.get(score, [])
            total_for_score = total_counts.get(score, 0)
            score_class = f"score-{score}" if score != -1 else "score--1"
            score_label = "No Score" if score == -1 else f"Score {score}"
            score_id = f"{iteration_name}-{'no-response' if score == -1 else str(score)}"
            
            # Calculate percentage
            percentage = (total_for_score / total_all_examples * 100) if total_all_examples > 0 else 0
            
            html_content += f"""
            <div class="score-section">
                <div class="score-header {score_class} collapse-header" data-target="score-content-{score_id}">
                    <span class="collapse-arrow" id="score-arrow-{score_id}"></span>
                    {score_label} - {total_for_score} examples ({percentage:.1f}%), showing {len(examples)}
                </div>
                <div class="collapsible-content" id="score-content-{score_id}">
"""
            
            if examples:
                for j, example in enumerate(examples):
                    html_content += f"""
                <div class="example">
                    <div class="example-header collapse-header" data-target="example-content-{score_id}-{j}">
                        <span class="collapse-arrow" id="example-arrow-{score_id}-{j}"></span>
                        Example {j+1}
                    </div>
                    <div class="collapsible-content" id="example-content-{score_id}-{j}">
                        <div class="prompt">
                            <div class="collapse-header" data-target="prompt-content-{score_id}-{j}">
                                <span class="collapse-arrow" id="prompt-arrow-{score_id}-{j}"></span>
                                <strong>User Prompt:</strong>
                            </div>
                            <div class="collapsible-content" id="prompt-content-{score_id}-{j}">
                                <div class="content-text">{escape_html(example.get('user_prompt', 'No prompt available'))}</div>
                            </div>
                        </div>
"""
                    
                    # Add reasoning first if available
                    if example.get('reasoning'):
                        html_content += f"""
                        <div class="reasoning">
                            <div class="collapse-header" data-target="reasoning-content-{score_id}-{j}">
                                <span class="collapse-arrow" id="reasoning-arrow-{score_id}-{j}"></span>
                                <strong>Reasoning:</strong>
                            </div>
                            <div class="collapsible-content" id="reasoning-content-{score_id}-{j}">
                                <div class="content-text">{escape_html(example.get('reasoning', ''))}</div>
                            </div>
                        </div>
"""
                    
                    # Add response after reasoning
                    html_content += f"""
                        <div class="response">
                            <div class="collapse-header" data-target="response-content-{score_id}-{j}">
                                <span class="collapse-arrow" id="response-arrow-{score_id}-{j}"></span>
                                <strong>Response:</strong>
                            </div>
                            <div class="collapsible-content" id="response-content-{score_id}-{j}">
                                <div class="content-text">{escape_html(example.get('response_only', example.get('full_response', 'No response available')))}</div>
                            </div>
                        </div>
                        
                        <div class="metadata">
                            <strong>Example Index:</strong> {escape_html(str(example.get('example_index', 'N/A')))} | 
                            <strong>Model:</strong> {escape_html(os.path.basename(example.get('model', 'N/A')))} | 
                            <strong>Timestamp:</strong> {escape_html(str(example.get('timestamp', 'N/A')))}
                        </div>
                    </div>
                </div>
"""
            else:
                # Show message when no examples are available for this score
                html_content += f"""
                <div class="example">
                    <div class="example-header">
                        No examples available for this score level
                    </div>
                    <div class="content-text" style="text-align: center; color: #666; font-style: italic; padding: 20px;">
                        No responses received a score of {score if score != -1 else 'no response'} in this iteration.
                    </div>
                </div>
"""
            
            html_content += """
                </div>
            </div>
"""
        
        html_content += "        </div>\n"  # Close tab content
    
    # Add JavaScript for tab switching and expand/collapse functionality
    html_content += """
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            const selectedTab = document.getElementById(tabName);
            if (selectedTab) {
                selectedTab.classList.add('active');
            }
            
            // Add active class to clicked tab button
            const clickedTab = document.querySelector(`[onclick="showTab('${tabName}')"]`);
            if (clickedTab) {
                clickedTab.classList.add('active');
            }
        }
        
        function toggleCollapse(elementId) {
            const content = document.getElementById(elementId);
            if (!content) {
                console.error('Content element not found:', elementId);
                return;
            }
            
            // Find the corresponding arrow element
            const arrow = document.getElementById(elementId.replace('content', 'arrow'));
            if (!arrow) {
                console.error('Arrow element not found for:', elementId);
                return;
            }
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                arrow.classList.remove('collapsed');
            } else {
                content.classList.add('collapsed');
                arrow.classList.add('collapsed');
            }
        }
        
        // Initialize only score sections as collapsed, keep inner content expanded
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, setting up functionality...');
            
            // Only collapse score sections (outermost layer)
            const scoreSections = document.querySelectorAll('[id^="score-content-"]');
            const scoreArrows = document.querySelectorAll('[id^="score-arrow-"]');
            
            scoreSections.forEach(content => {
                content.classList.add('collapsed');
            });
            scoreArrows.forEach(arrow => {
                arrow.classList.add('collapsed');
            });
            
            // Keep all inner content (examples, prompts, responses, reasoning) expanded
            const innerContents = document.querySelectorAll('[id^="example-content-"], [id^="prompt-content-"], [id^="response-content-"], [id^="reasoning-content-"]');
            const innerArrows = document.querySelectorAll('[id^="example-arrow-"], [id^="prompt-arrow-"], [id^="response-arrow-"], [id^="reasoning-arrow-"]');
            
            innerContents.forEach(content => {
                content.classList.remove('collapsed');
            });
            innerArrows.forEach(arrow => {
                arrow.classList.remove('collapsed');
            });
            
            // Add click event listeners to all collapse headers
            const collapseHeaders = document.querySelectorAll('.collapse-header');
            console.log('Found', collapseHeaders.length, 'collapse headers');
            
            collapseHeaders.forEach(header => {
                header.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const targetId = this.getAttribute('data-target');
                    console.log('Clicked header with target:', targetId);
                    if (targetId) {
                        toggleCollapse(targetId);
                    }
                });
            });
        });
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML report from evaluation output')
    parser.add_argument('eval_dir', help='Path to evaluation output directory')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples per score (default: 5)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible sampling')
    parser.add_argument('--auto-generate-plots', action='store_true', 
                       help='Automatically generate missing distribution plots using the plotting script')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Automatically prepend evaluation_output if not already included
    eval_path = Path(args.eval_dir)
    if not eval_path.parts or eval_path.parts[0] != 'evaluation_output':
        eval_dir = str(Path('evaluation_output') / eval_path)
    else:
        eval_dir = args.eval_dir
    
    # Check if evaluation directory exists
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory '{eval_dir}' does not exist.")
        return
    
    # Generate output path based on input path (always remove evaluation_output prefix for output)
    eval_path = Path(eval_dir)
    if eval_path.parts and eval_path.parts[0] == 'evaluation_output':
        # Remove 'evaluation_output' and use the rest of the path
        relative_path = Path(*eval_path.parts[1:])
        output_path = Path('evaluation_reports') / relative_path / 'evaluation_report.html'
    else:
        # Fallback if path doesn't start with evaluation_output
        output_path = Path('evaluation_reports') / eval_path.name / 'evaluation_report.html'
    args.output = str(output_path)
    
    # Update args.eval_dir to use the resolved path
    args.eval_dir = eval_dir
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Find all evaluation combinations in the directory
    evaluation_combinations = find_all_evaluation_combinations(args.eval_dir)
    
    if evaluation_combinations:
        print(f"Found {len(evaluation_combinations)} evaluation combinations:")
        for combo in evaluation_combinations:
            relative_path = os.path.relpath(combo, args.eval_dir)
            print(f"  {relative_path}")
        
        print(f"\nGenerating reports for all combinations...")
        
        for combo_dir in evaluation_combinations:
            # Generate output path for this combination
            eval_path = Path(combo_dir)
            if eval_path.parts and eval_path.parts[0] == 'evaluation_output':
                # Remove 'evaluation_output' and use the rest of the path
                relative_path = Path(*eval_path.parts[1:])
                output_path = Path('evaluation_reports') / relative_path / 'evaluation_report.html'
            else:
                # Fallback if path doesn't start with evaluation_output
                output_path = Path('evaluation_reports') / eval_path.name / 'evaluation_report.html'
            
            combo_output = str(output_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(combo_output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            print(f"\nProcessing: {os.path.relpath(combo_dir, args.eval_dir)}")
            
            # Get iteration directories for this combination
            iteration_dirs = find_iteration_directories(combo_dir)
            
            if iteration_dirs:
                print(f"  Found {len(iteration_dirs)} iteration directories.")
                
                iteration_data = {}
                
                for iteration_dir in iteration_dirs:
                    iteration_name = os.path.basename(iteration_dir)
                    print(f"    Processing {iteration_name}...")
                    
                    # Load and process data for this iteration
                    data = load_evaluation_data(iteration_dir)
                    
                    if not data:
                        print(f"    No evaluation data found in {iteration_name}.")
                        continue
                    
                    print(f"    Loaded {len(data)} evaluation examples from {iteration_name}")
                    
                    # Categorize by score
                    categorized, total_counts = categorize_by_score(data, args.num_examples)
                    
                    iteration_data[iteration_name] = (categorized, total_counts)
                
                if iteration_data:
                    # Extract experiment info and generate single tabbed HTML report
                    experiment_info = extract_experiment_info(combo_dir)
                    
                    # Check if we need to auto-generate plots for this combination
                    if args.auto_generate_plots:
                        print(f"  Checking for distribution plots...")
                        auto_generate_missing_plots(experiment_info, args.eval_dir, True)
                    
                    generate_tabbed_html_report(iteration_data, combo_output, experiment_info, args.auto_generate_plots, args.eval_dir)
                    print(f"  Report generated: {combo_output}")
                else:
                    print(f"  No data found in any iteration directories for this combination.")
            else:
                print(f"  No iteration directories found in this combination.")
        
        print(f"\nAll reports generated successfully!")
    else:
        # Check if this directory itself contains iteration subdirectories
        iteration_dirs = find_iteration_directories(args.eval_dir)
        
        if iteration_dirs:
            print(f"Found {len(iteration_dirs)} iteration directories. Generating tabbed report with all iterations.")
            
            iteration_data = {}
            
            for iteration_dir in iteration_dirs:
                iteration_name = os.path.basename(iteration_dir)
                print(f"Processing {iteration_name}...")
                
                # Load and process data for this iteration
                data = load_evaluation_data(iteration_dir)
                
                if not data:
                    print(f"No evaluation data found in {iteration_name}.")
                    continue
                
                print(f"Loaded {len(data)} evaluation examples from {iteration_name}")
                
                # Categorize by score
                categorized, total_counts = categorize_by_score(data, args.num_examples)
                
                print(f"Examples per score in {iteration_name}:")
                for score in sorted(categorized.keys()):
                    count = len(categorized[score])
                    total_count = total_counts[score]
                    print(f"  Score {score}: {total_count} total examples, showing {count}")
                
                iteration_data[iteration_name] = (categorized, total_counts)
            
            if iteration_data:
                # Extract experiment info and generate single tabbed HTML report
                experiment_info = extract_experiment_info(args.eval_dir)
                generate_tabbed_html_report(iteration_data, args.output, experiment_info, args.auto_generate_plots, args.eval_dir)
                print(f"\nTabbed report generated successfully!")
                print(f"Open {args.output} in your web browser to view the report.")
            else:
                print("No data found in any iteration directories.")
        else:
            # No iteration directories found, process as single directory
            print(f"No iteration directories found. Processing as single evaluation directory.")
            print(f"Loading evaluation data from: {args.eval_dir}")
            data = load_evaluation_data(args.eval_dir)
            
            if not data:
                print("No evaluation data found.")
                return
            
            print(f"Loaded {len(data)} evaluation examples")
            
            # Categorize by score
            categorized, total_counts = categorize_by_score(data, args.num_examples)
            
            print("Examples per score:")
            for score in sorted(categorized.keys()):
                count = len(categorized[score])
                total_count = total_counts[score]
                print(f"  Score {score}: {total_count} total examples, showing {count}")
            
            # Generate single iteration HTML report
            iteration_data = {"single": (categorized, total_counts)}
            generate_tabbed_html_report(iteration_data, args.output, auto_generate_plots=args.auto_generate_plots, eval_dir=args.eval_dir)
            
            print(f"\nReport generated successfully!")
            print(f"Open {args.output} in your web browser to view the report.")


if __name__ == "__main__":
    main()
