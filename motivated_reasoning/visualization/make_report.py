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


def generate_html_report(categorized_data: Dict[int, List[Dict[str, Any]]], total_counts: Dict[int, int], output_file: str):
    """Generate an HTML report from the categorized data."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Score Examples Report</title>
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
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        .score-section {{
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .score-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.5em;
            font-weight: bold;
        }}
        .score-1 {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }}
        .score-2 {{ background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%); }}
        .score-3 {{ background: linear-gradient(135deg, #ffd54f 0%, #ffc107 100%); }}
        .score-4 {{ background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%); }}
        .score-5 {{ background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%); }}
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Score Examples Report</h1>
        
"""
    

    
    # Calculate total examples across all scores for percentage calculation
    total_all_examples = sum(total_counts.values())
    
    # Add examples for each score
    for score in sorted(categorized_data.keys(), reverse=True):
        examples = categorized_data[score]
        total_for_score = total_counts[score]
        score_class = f"score-{score}" if score != -1 else "score--1"
        score_label = "No Response" if score == -1 else f"Score {score}"
        score_id = "no-response" if score == -1 else str(score)
        
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
        
        for i, example in enumerate(examples):
            html_content += f"""
            <div class="example">
                <div class="example-header collapse-header" data-target="example-content-{score_id}-{i}">
                    <span class="collapse-arrow" id="example-arrow-{score_id}-{i}"></span>
                    Example {i+1}
                </div>
                <div class="collapsible-content" id="example-content-{score_id}-{i}">
                    <div class="prompt">
                        <div class="collapse-header" data-target="prompt-content-{score_id}-{i}">
                            <span class="collapse-arrow" id="prompt-arrow-{score_id}-{i}"></span>
                            <strong>User Prompt:</strong>
                        </div>
                        <div class="collapsible-content" id="prompt-content-{score_id}-{i}">
                            <div class="content-text">{example.get('user_prompt', 'No prompt available')}</div>
                        </div>
                    </div>
"""
            
            # Add reasoning first if available
            if example.get('reasoning'):
                html_content += f"""
                    <div class="reasoning">
                        <div class="collapse-header" data-target="reasoning-content-{score_id}-{i}">
                            <span class="collapse-arrow" id="reasoning-arrow-{score_id}-{i}"></span>
                            <strong>Reasoning:</strong>
                        </div>
                        <div class="collapsible-content" id="reasoning-content-{score_id}-{i}">
                            <div class="content-text">{example.get('reasoning', '')}</div>
                        </div>
                    </div>
"""
            
            # Add response after reasoning
            html_content += f"""
                    <div class="response">
                        <div class="collapse-header" data-target="response-content-{score_id}-{i}">
                            <span class="collapse-arrow" id="response-arrow-{score_id}-{i}"></span>
                            <strong>Response:</strong>
                        </div>
                        <div class="collapsible-content" id="response-content-{score_id}-{i}">
                            <div class="content-text">{example.get('response_only', example.get('full_response', 'No response available'))}</div>
                        </div>
                    </div>
                    
                    <div class="metadata">
                        <strong>Example Index:</strong> {example.get('example_index', 'N/A')} | 
                        <strong>Model:</strong> {os.path.basename(example.get('model', 'N/A'))} | 
                        <strong>Timestamp:</strong> {example.get('timestamp', 'N/A')}
                    </div>
                </div>
            </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    # Add JavaScript for expand/collapse functionality
    html_content += """
    </div>
    
    <script>
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
        
        // Initialize all content as expanded
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, setting up collapse functionality...');
            
            const collapsibleContents = document.querySelectorAll('.collapsible-content');
            collapsibleContents.forEach(content => {
                content.classList.remove('collapsed');
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
    parser.add_argument('--output', '-o', default=None, 
                       help='Output HTML file path (default: auto-generated based on input path)')
    parser.add_argument('--examples', '-e', type=int, default=5,
                       help='Number of examples per score (default: 5)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Check if evaluation directory exists
    if not os.path.exists(args.eval_dir):
        print(f"Error: Evaluation directory '{args.eval_dir}' does not exist.")
        return
    
    # Generate output path if not provided
    if args.output is None:
        # Extract the relative path from evaluation_output
        eval_path = Path(args.eval_dir)
        if eval_path.parts and eval_path.parts[0] == 'evaluation_output':
            # Remove 'evaluation_output' and use the rest of the path
            relative_path = Path(*eval_path.parts[1:])
            output_path = Path('evaluation_reports') / relative_path / 'evaluation_report.html'
        else:
            # Fallback if path doesn't start with evaluation_output
            output_path = Path('evaluation_reports') / eval_path.name / 'evaluation_report.html'
        args.output = str(output_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    print(f"Loading evaluation data from: {args.eval_dir}")
    data = load_evaluation_data(args.eval_dir)
    
    if not data:
        print("No evaluation data found.")
        return
    
    print(f"Loaded {len(data)} evaluation examples")
    
    # Categorize by score
    categorized, total_counts = categorize_by_score(data, args.examples)
    
    print("Examples per score:")
    for score in sorted(categorized.keys()):
        count = len(categorized[score])
        total_count = total_counts[score]
        print(f"  Score {score}: {total_count} total examples, showing {count}")
    
    # Generate HTML report
    generate_html_report(categorized, total_counts, args.output)
    
    print(f"\nReport generated successfully!")
    print(f"Open {args.output} in your web browser to view the report.")


if __name__ == "__main__":
    main()
