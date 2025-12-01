"""
Data dicing script for motivated reasoning evaluation results.

This script analyzes evaluation results to understand which datapoints that disagree 
with the constitution (wrong answers) are caught at different levels of motivated 
reasoning detection.

Usage:
    python study_evaluation_results.py evaluation_dir [--output-dir OUTPUT_DIR] [options]

Example:
    python study_evaluation_results.py now-09_20_201429
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
from typing import Dict, List, Optional, Tuple, Any


def discover_experiments(evaluation_dir: str) -> List[Dict[str, str]]:
    """
    Discover all experiment configurations in the evaluation directory.
    
    Args:
        evaluation_dir: Name of the evaluation directory
        
    Returns:
        List of experiment configurations with prompt_type, evaluator, suffix, iteration
    """
    evaluation_path = Path("evaluation_output") / evaluation_dir
    if not evaluation_path.exists():
        print(f"Error: Evaluation directory {evaluation_path} does not exist")
        return []
    
    experiments = []
    
    # Walk through prompt types
    for prompt_type_dir in evaluation_path.iterdir():
        if not prompt_type_dir.is_dir():
            continue
            
        prompt_type = prompt_type_dir.name
        
        # Look for reasoning evaluations (these have the suffix structure we need)
        reasoning_dir = prompt_type_dir / "reasoning"
        if not reasoning_dir.exists():
            continue
            
        # Find evaluators
        for evaluator_dir in reasoning_dir.iterdir():
            if not evaluator_dir.is_dir() or not evaluator_dir.name.startswith("evaluator-"):
                continue
                
            evaluator_name = evaluator_dir.name.replace("evaluator-", "")
            
            # Find suffixes
            for suffix_dir in evaluator_dir.iterdir():
                if not suffix_dir.is_dir():
                    continue
                    
                suffix_name = suffix_dir.name
                
                # Find iterations
                for iteration_dir in suffix_dir.iterdir():
                    if not iteration_dir.is_dir() or not iteration_dir.name.startswith("iteration-"):
                        continue
                        
                    iteration = iteration_dir.name.replace("iteration-", "")
                    
                    # Check if there are JSON files
                    json_files = list(iteration_dir.glob("*eval*.json"))
                    if json_files:
                        experiments.append({
                            'prompt_type': prompt_type,
                            'evaluator': evaluator_name,
                            'suffix': suffix_name,
                            'iteration': iteration
                        })
    
    print(f"Discovered {len(experiments)} experiment configurations")
    return experiments


def load_correctness_data(evaluation_dir: str, prompt_type: str, evaluator: str, iteration: str) -> Dict[int, Dict]:
    """
    Load correctness evaluation data (response scores).
    Always uses flash-lite evaluator for correctness judgments regardless of which evaluator
    is being used for reasoning evaluation.
    
    Args:
        evaluation_dir: Name of the evaluation directory
        prompt_type: Prompt type name
        evaluator: Evaluator name (ignored - always uses flash-lite)
        iteration: Iteration identifier
        
    Returns:
        Dictionary mapping example_index to correctness data
    """
    # Always use flash-lite evaluator for correctness data
    correctness_path = (Path("evaluation_output") / evaluation_dir / prompt_type / 
                       "response" / "evaluator-gemini-25-flash-lite" / "simple_correct_choice" / 
                       f"iteration-{iteration}")
    
    if not correctness_path.exists():
        print(f"Warning: Correctness path does not exist: {correctness_path}")
        return {}
    
    # Find the most recent JSON file
    json_files = list(correctness_path.glob("*eval*.json"))
    if not json_files:
        print(f"Warning: No evaluation JSON files found in {correctness_path}")
        return {}
    
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Create lookup by example_index
        correctness_lookup = {}
        for item in data:
            example_index = item.get('example_index')
            if example_index is not None:
                correctness_lookup[example_index] = item
        
        print(f"  Loaded correctness data: {len(correctness_lookup)} datapoints from {latest_file.name}")
        return correctness_lookup
        
    except Exception as e:
        print(f"Error loading correctness data from {latest_file}: {e}")
        return {}


def load_reasoning_data(evaluation_dir: str, prompt_type: str, evaluator: str, 
                       suffix: str, iteration: str) -> Dict[int, Dict]:
    """
    Load reasoning evaluation data (motivated reasoning scores).
    
    Args:
        evaluation_dir: Name of the evaluation directory
        prompt_type: Prompt type name
        evaluator: Evaluator name
        suffix: Suffix name
        iteration: Iteration identifier
        
    Returns:
        Dictionary mapping example_index to reasoning data
    """
    reasoning_path = (Path("evaluation_output") / evaluation_dir / prompt_type / 
                     "reasoning" / f"evaluator-{evaluator}" / suffix / f"iteration-{iteration}")
    
    if not reasoning_path.exists():
        print(f"Warning: Reasoning path does not exist: {reasoning_path}")
        return {}
    
    # Find the most recent JSON file
    json_files = list(reasoning_path.glob("*eval*.json"))
    if not json_files:
        print(f"Warning: No evaluation JSON files found in {reasoning_path}")
        return {}
    
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Create lookup by example_index
        reasoning_lookup = {}
        for item in data:
            example_index = item.get('example_index')
            if example_index is not None:
                reasoning_lookup[example_index] = item
        
        print(f"  Loaded reasoning data: {len(reasoning_lookup)} datapoints from {latest_file.name}")
        return reasoning_lookup
        
    except Exception as e:
        print(f"Error loading reasoning data from {latest_file}: {e}")
        return {}


def match_and_categorize_datapoints(correctness_data: Dict[int, Dict], 
                                  reasoning_data: Dict[int, Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Match datapoints by example_index and categorize by correctness and reasoning level.
    
    Args:
        correctness_data: Dictionary mapping example_index to correctness data
        reasoning_data: Dictionary mapping example_index to reasoning data
        
    Returns:
        Tuple of (categorized_data, skipped_datapoints)
        categorized_data: {'wrong': {level: [datapoints]}, 'correct': {level: [datapoints]}}
        skipped_datapoints: List of datapoints with missing/invalid correctness scores
    """
    categorized = {
        'wrong': defaultdict(list),
        'correct': defaultdict(list)
    }
    skipped = []
    
    # Get all unique example indices
    all_indices = set(correctness_data.keys()) | set(reasoning_data.keys())
    
    for example_index in all_indices:
        correctness_info = correctness_data.get(example_index, {})
        reasoning_info = reasoning_data.get(example_index, {})
        
        # Check if we have valid correctness information
        response_score = correctness_info.get('evaluator_score')
        if response_score not in [0, 1]:
            # Skip datapoints without clear correctness information
            skipped_datapoint = {
                'example_index': example_index,
                'reason': 'missing_or_invalid_correctness_score',
                'response_evaluation_score': response_score,
                'has_correctness_data': bool(correctness_info),
                'has_reasoning_data': bool(reasoning_info)
            }
            skipped.append(skipped_datapoint)
            continue
        
        # Determine correctness category
        correctness_category = 'correct' if response_score == 1 else 'wrong'
        
        # Determine reasoning level
        reasoning_score = reasoning_info.get('reasoning_evaluation_score')
        if reasoning_score is None or reasoning_score == -1:
            reasoning_level = 'no_reasoning_score'
        elif reasoning_score in [1, 2, 3, 4, 5]:
            reasoning_level = reasoning_score
        else:
            reasoning_level = 'invalid_reasoning_score'
        
        # Create enriched datapoint
        enriched_datapoint = {
            'example_index': example_index,
            'correctness_category': correctness_category,
            'reasoning_level': reasoning_level,
            'correctness_data': correctness_info,
            'reasoning_data': reasoning_info
        }
        
        categorized[correctness_category][reasoning_level].append(enriched_datapoint)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    categorized = {
        'wrong': dict(categorized['wrong']),
        'correct': dict(categorized['correct'])
    }
    
    return categorized, skipped


def save_categorized_data(categorized_data: Dict, skipped_data: List[Dict], 
                         output_path: Path, config: Dict[str, str]) -> None:
    """
    Save categorized data to organized directory structure.
    
    Args:
        categorized_data: Dictionary with 'wrong' and 'correct' categories
        skipped_data: List of skipped datapoints
        output_path: Base output path for this experiment
        config: Experiment configuration dict
    """
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    wrong_dir = output_path / "wrong_datapoints"
    correct_dir = output_path / "correct_datapoints"
    wrong_dir.mkdir(exist_ok=True)
    correct_dir.mkdir(exist_ok=True)
    
    # Save wrong datapoints by reasoning level
    total_wrong = 0
    for level, datapoints in categorized_data['wrong'].items():
        if not datapoints:
            continue
            
        if level == 'no_reasoning_score':
            filename = "no_reasoning_score.jsonl"
        elif level == 'invalid_reasoning_score':
            filename = "invalid_reasoning_score.jsonl"
        else:
            level_names = {
                1: "level_1_genuine",
                2: "level_2_mostly_genuine", 
                3: "level_3_mixed",
                4: "level_4_mostly_motivated",
                5: "level_5_fully_motivated"
            }
            filename = f"{level_names.get(level, f'level_{level}')}.jsonl"
        
        filepath = wrong_dir / filename
        with open(filepath, 'w') as f:
            for datapoint in datapoints:
                f.write(json.dumps(datapoint, indent=2) + '\n')
        
        total_wrong += len(datapoints)
        print(f"    Saved {len(datapoints)} wrong datapoints to {filename}")
    
    # Save correct datapoints by reasoning level
    total_correct = 0
    for level, datapoints in categorized_data['correct'].items():
        if not datapoints:
            continue
            
        if level == 'no_reasoning_score':
            filename = "no_reasoning_score.jsonl"
        elif level == 'invalid_reasoning_score':
            filename = "invalid_reasoning_score.jsonl"
        else:
            level_names = {
                1: "level_1_genuine",
                2: "level_2_mostly_genuine",
                3: "level_3_mixed", 
                4: "level_4_mostly_motivated",
                5: "level_5_fully_motivated"
            }
            filename = f"{level_names.get(level, f'level_{level}')}.jsonl"
        
        filepath = correct_dir / filename
        with open(filepath, 'w') as f:
            for datapoint in datapoints:
                f.write(json.dumps(datapoint, indent=2) + '\n')
        
        total_correct += len(datapoints)
        print(f"    Saved {len(datapoints)} correct datapoints to {filename}")
    
    # Save skipped datapoints
    if skipped_data:
        skipped_path = output_path / "skipped_datapoints.json"
        with open(skipped_path, 'w') as f:
            json.dump(skipped_data, f, indent=2)
        print(f"    Saved {len(skipped_data)} skipped datapoints to skipped_datapoints.json")
    
    # Generate and save summary statistics
    summary = generate_summary_stats(categorized_data, skipped_data, config)
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"    Summary: {total_wrong} wrong, {total_correct} correct, {len(skipped_data)} skipped")


def generate_summary_stats(categorized_data: Dict, skipped_data: List[Dict], 
                          config: Dict[str, str]) -> Dict:
    """
    Generate summary statistics for the categorized data.
    
    Args:
        categorized_data: Dictionary with categorized datapoints
        skipped_data: List of skipped datapoints
        config: Experiment configuration
        
    Returns:
        Summary statistics dictionary
    """
    def count_by_level(category_data):
        counts = {}
        total = 0
        for level, datapoints in category_data.items():
            count = len(datapoints)
            counts[str(level)] = count
            total += count
        counts['total'] = total
        return counts
    
    summary = {
        'experiment_config': config,
        'statistics': {
            'wrong_datapoints': count_by_level(categorized_data['wrong']),
            'correct_datapoints': count_by_level(categorized_data['correct']),
            'skipped_datapoints': len(skipped_data)
        }
    }
    
    # Add overall totals
    total_processed = (summary['statistics']['wrong_datapoints']['total'] + 
                      summary['statistics']['correct_datapoints']['total'])
    summary['statistics']['total_processed'] = total_processed
    summary['statistics']['total_with_skipped'] = total_processed + len(skipped_data)
    
    return summary


def process_experiment(config: Dict[str, str], output_base: str) -> bool:
    """
    Process a single experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
        output_base: Base output directory
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {config['prompt_type']}/{config['evaluator']}/{config['suffix']}/iteration-{config['iteration']}")
    
    # Load correctness data
    correctness_data = load_correctness_data(
        config['evaluation_dir'], config['prompt_type'], 
        config['evaluator'], config['iteration']
    )
    
    # Load reasoning data
    reasoning_data = load_reasoning_data(
        config['evaluation_dir'], config['prompt_type'], 
        config['evaluator'], config['suffix'], config['iteration']
    )
    
    if not correctness_data and not reasoning_data:
        print("  No data found for this configuration, skipping")
        return False
    
    # Match and categorize datapoints
    categorized_data, skipped_data = match_and_categorize_datapoints(
        correctness_data, reasoning_data
    )
    
    # Create output path
    output_path = (Path(output_base) / config['evaluation_dir'] / config['prompt_type'] / 
                  f"evaluator-{config['evaluator']}" / config['suffix'] / f"iteration-{config['iteration']}")
    
    # Save results
    save_categorized_data(categorized_data, skipped_data, output_path, config)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Analyze motivated reasoning evaluation results')
    parser.add_argument('evaluation_dir', type=str, 
                       help='Evaluation directory name (e.g., now-09_20_201429)')
    parser.add_argument('--output-dir', default='analysis_output', 
                       help='Output directory (default: analysis_output)')
    parser.add_argument('--prompt-type', type=str, 
                       help='Specific prompt type to process (optional)')
    parser.add_argument('--evaluator', type=str, 
                       help='Specific evaluator to process (optional)')
    parser.add_argument('--suffix', type=str, 
                       help='Specific suffix to process (optional)')
    parser.add_argument('--iteration', type=str, 
                       help='Specific iteration to process (optional)')
    
    args = parser.parse_args()
    
    print(f"Analyzing evaluation results from: {args.evaluation_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Discover all experiments
    experiments = discover_experiments(args.evaluation_dir)
    if not experiments:
        print("No experiments found!")
        return
    
    # Filter experiments based on arguments
    filtered_experiments = []
    for exp in experiments:
        exp['evaluation_dir'] = args.evaluation_dir  # Add for convenience
        
        if args.prompt_type and exp['prompt_type'] != args.prompt_type:
            continue
        if args.evaluator and exp['evaluator'] != args.evaluator:
            continue
        if args.suffix and exp['suffix'] != args.suffix:
            continue
        if args.iteration and exp['iteration'] != args.iteration:
            continue
        
        filtered_experiments.append(exp)
    
    print(f"Processing {len(filtered_experiments)} experiment configurations")
    
    # Process each experiment
    successful = 0
    failed = 0
    
    for i, config in enumerate(filtered_experiments):
        print(f"\n[{i+1}/{len(filtered_experiments)}]", end=" ")
        
        try:
            if process_experiment(config, args.output_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing experiment: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
