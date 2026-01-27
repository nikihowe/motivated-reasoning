"""
Data dicing script for reasonableness evaluation results.

This script analyzes evaluation results to understand the relationship between
correctness (whether the model gave the correct answer according to constitution)
and reasonableness (whether the evaluator deemed the answer reasonable).

The script creates a 2x2 categorization:
- Correct & Reasonable: Model gave correct answer AND evaluator deemed it reasonable
- Correct & Unreasonable: Model gave correct answer BUT evaluator deemed it unreasonable
- Incorrect & Reasonable: Model gave wrong answer BUT evaluator deemed it reasonable
- Incorrect & Unreasonable: Model gave wrong answer AND evaluator deemed it unreasonable

Usage:
    python study_reasonabless_differences.py evaluation_dir [--output-dir OUTPUT_DIR] [options]

Example:
    python study_reasonabless_differences.py now-09_20_201429
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
from typing import Dict, List, Optional, Tuple, Any


def discover_experiments(evaluation_dir: str) -> List[Dict[str, str]]:
    """
    Discover all experiment configurations with reasonableness evaluations.

    Args:
        evaluation_dir: Name of the evaluation directory

    Returns:
        List of experiment configurations with prompt_type, eval_target, evaluator, iteration
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

        # Walk through eval targets (constitution_and_response, system_and_response, etc.)
        for eval_target_dir in prompt_type_dir.iterdir():
            if not eval_target_dir.is_dir():
                continue

            eval_target = eval_target_dir.name

            # Find evaluators
            for evaluator_dir in eval_target_dir.iterdir():
                if not evaluator_dir.is_dir() or not evaluator_dir.name.startswith("evaluator-"):
                    continue

                evaluator_name = evaluator_dir.name.replace("evaluator-", "")

                # Look for simple_reasonable_recommendation_v2 directories
                reasonableness_dir = evaluator_dir / "simple_reasonable_recommendation_v2"
                if not reasonableness_dir.exists():
                    continue

                # Find iterations
                for iteration_dir in reasonableness_dir.iterdir():
                    if not iteration_dir.is_dir() or not iteration_dir.name.startswith("iteration-"):
                        continue

                    iteration = iteration_dir.name.replace("iteration-", "")

                    # Check if there are JSON files
                    json_files = list(iteration_dir.glob("*eval*.json"))
                    if json_files:
                        experiments.append({
                            'prompt_type': prompt_type,
                            'eval_target': eval_target,
                            'evaluator': evaluator_name,
                            'iteration': iteration
                        })

    print(f"Discovered {len(experiments)} experiment configurations with reasonableness evaluations")
    return experiments


def load_correctness_data(evaluation_dir: str, prompt_type: str, evaluator: str, iteration: str) -> Dict[int, Dict]:
    """
    Load correctness evaluation data (response scores).
    Always uses flash-lite evaluator for correctness judgments regardless of which evaluator
    is being used for reasonableness evaluation.

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


def load_reasonableness_data(evaluation_dir: str, prompt_type: str, eval_target: str,
                           evaluator: str, iteration: str) -> Dict[int, Dict]:
    """
    Load reasonableness evaluation data (simple_reasonable_recommendation_v2 scores).

    Args:
        evaluation_dir: Name of the evaluation directory
        prompt_type: Prompt type name
        eval_target: Eval target name (e.g., system_and_response, constitution_and_response)
        evaluator: Evaluator name
        iteration: Iteration identifier

    Returns:
        Dictionary mapping example_index to reasonableness data
    """
    reasonableness_path = (Path("evaluation_output") / evaluation_dir / prompt_type / eval_target /
                          f"evaluator-{evaluator}" / "simple_reasonable_recommendation_v2" /
                          f"iteration-{iteration}")

    if not reasonableness_path.exists():
        print(f"Warning: Reasonableness path does not exist: {reasonableness_path}")
        return {}

    # Find the most recent JSON file
    json_files = list(reasonableness_path.glob("*eval*.json"))
    if not json_files:
        print(f"Warning: No evaluation JSON files found in {reasonableness_path}")
        return {}

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Create lookup by example_index
        reasonableness_lookup = {}
        for item in data:
            example_index = item.get('example_index')
            if example_index is not None:
                reasonableness_lookup[example_index] = item

        print(f"  Loaded reasonableness data: {len(reasonableness_lookup)} datapoints from {latest_file.name}")
        return reasonableness_lookup

    except Exception as e:
        print(f"Error loading reasonableness data from {latest_file}: {e}")
        return {}


def match_and_categorize_datapoints(correctness_data: Dict[int, Dict],
                                  reasonableness_data: Dict[int, Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Match datapoints by example_index and categorize by correctness and reasonableness.

    Args:
        correctness_data: Dictionary mapping example_index to correctness data
        reasonableness_data: Dictionary mapping example_index to reasonableness data

    Returns:
        Tuple of (categorized_data, skipped_datapoints)
        categorized_data: {
            'correct_reasonable': [datapoints],
            'correct_unreasonable': [datapoints],
            'incorrect_reasonable': [datapoints],
            'incorrect_unreasonable': [datapoints]
        }
        skipped_datapoints: List of datapoints with missing/invalid scores
    """
    categorized = {
        'correct_reasonable': [],
        'correct_unreasonable': [],
        'incorrect_reasonable': [],
        'incorrect_unreasonable': []
    }
    skipped = []

    # Get all unique example indices
    all_indices = set(correctness_data.keys()) | set(reasonableness_data.keys())

    for example_index in all_indices:
        correctness_info = correctness_data.get(example_index, {})
        reasonableness_info = reasonableness_data.get(example_index, {})

        # Check if we have valid correctness information
        correctness_score = correctness_info.get('evaluator_score')
        if correctness_score not in [0, 1]:
            # Skip datapoints without clear correctness information
            skipped_datapoint = {
                'example_index': example_index,
                'reason': 'missing_or_invalid_correctness_score',
                'correctness_score': correctness_score,
                'has_correctness_data': bool(correctness_info),
                'has_reasonableness_data': bool(reasonableness_info)
            }
            skipped.append(skipped_datapoint)
            continue

        # Check if we have valid reasonableness information
        reasonableness_score = reasonableness_info.get('evaluator_score')
        if reasonableness_score not in [0, 1]:
            # Skip datapoints without clear reasonableness information
            skipped_datapoint = {
                'example_index': example_index,
                'reason': 'missing_or_invalid_reasonableness_score',
                'reasonableness_score': reasonableness_score,
                'has_correctness_data': bool(correctness_info),
                'has_reasonableness_data': bool(reasonableness_info)
            }
            skipped.append(skipped_datapoint)
            continue

        # Determine category based on both scores
        if correctness_score == 1 and reasonableness_score == 1:
            category = 'correct_reasonable'
        elif correctness_score == 1 and reasonableness_score == 0:
            category = 'correct_unreasonable'
        elif correctness_score == 0 and reasonableness_score == 1:
            category = 'incorrect_reasonable'
        else:  # correctness_score == 0 and reasonableness_score == 0
            category = 'incorrect_unreasonable'

        # Create enriched datapoint
        enriched_datapoint = {
            'example_index': example_index,
            'category': category,
            'correctness_score': correctness_score,
            'reasonableness_score': reasonableness_score,
            'correctness_data': correctness_info,
            'reasonableness_data': reasonableness_info
        }

        categorized[category].append(enriched_datapoint)

    return categorized, skipped


def save_categorized_data(categorized_data: Dict, skipped_data: List[Dict],
                         output_path: Path, config: Dict[str, str]) -> None:
    """
    Save categorized data to organized directory structure.

    Args:
        categorized_data: Dictionary with four categories of datapoints
        skipped_data: List of skipped datapoints
        output_path: Base output path for this experiment
        config: Experiment configuration dict
    """
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each category
    for category, datapoints in categorized_data.items():
        if not datapoints:
            continue

        filepath = output_path / f"{category}.jsonl"
        with open(filepath, 'w') as f:
            for datapoint in datapoints:
                f.write(json.dumps(datapoint, indent=2) + '\n')

        print(f"    Saved {len(datapoints)} datapoints to {category}.jsonl")

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

    # Print summary
    total_processed = sum(len(datapoints) for datapoints in categorized_data.values())
    print(f"    Summary: {total_processed} total, {len(skipped_data)} skipped")
    for category, datapoints in categorized_data.items():
        if datapoints:
            print(f"      {category}: {len(datapoints)}")


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
    category_counts = {}
    total_processed = 0

    for category, datapoints in categorized_data.items():
        count = len(datapoints)
        category_counts[category] = count
        total_processed += count

    summary = {
        'experiment_config': config,
        'statistics': {
            'category_counts': category_counts,
            'total_processed': total_processed,
            'skipped_datapoints': len(skipped_data),
            'total_with_skipped': total_processed + len(skipped_data)
        }
    }

    # Add some analysis metrics
    if total_processed > 0:
        correct_total = category_counts.get('correct_reasonable', 0) + category_counts.get('correct_unreasonable', 0)
        reasonable_total = category_counts.get('correct_reasonable', 0) + category_counts.get('incorrect_reasonable', 0)

        summary['analysis'] = {
            'correctness_rate': correct_total / total_processed,
            'reasonableness_rate': reasonable_total / total_processed,
            'agreement_rate': (category_counts.get('correct_reasonable', 0) +
                              category_counts.get('incorrect_unreasonable', 0)) / total_processed
        }

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
    print(f"\nProcessing: {config['prompt_type']}/{config['eval_target']}/{config['evaluator']}/iteration-{config['iteration']}")

    # Load correctness data
    correctness_data = load_correctness_data(
        config['evaluation_dir'], config['prompt_type'],
        config['evaluator'], config['iteration']
    )

    # Load reasonableness data
    reasonableness_data = load_reasonableness_data(
        config['evaluation_dir'], config['prompt_type'], config['eval_target'],
        config['evaluator'], config['iteration']
    )

    if not correctness_data and not reasonableness_data:
        print("  No data found for this configuration, skipping")
        return False

    # Match and categorize datapoints
    categorized_data, skipped_data = match_and_categorize_datapoints(
        correctness_data, reasonableness_data
    )

    # Create output path
    output_path = (Path(output_base) / config['evaluation_dir'] / config['prompt_type'] /
                  config['eval_target'] / f"evaluator-{config['evaluator']}" / f"iteration-{config['iteration']}")

    # Save results
    save_categorized_data(categorized_data, skipped_data, output_path, config)

    return True


def main():
    parser = argparse.ArgumentParser(description='Analyze reasonableness evaluation results')
    parser.add_argument('evaluation_dir', type=str,
                       help='Evaluation directory name (e.g., now-09_20_201429)')
    parser.add_argument('--output-dir', default='analysis_output/reasonableness',
                       help='Output directory (default: analysis_output/reasonableness)')
    parser.add_argument('--prompt-type', type=str,
                       help='Specific prompt type to process (optional)')
    parser.add_argument('--eval-target', type=str,
                       help='Specific eval target to process (optional)')
    parser.add_argument('--evaluator', type=str,
                       help='Specific evaluator to process (optional)')
    parser.add_argument('--iteration', type=str,
                       help='Specific iteration to process (optional)')

    args = parser.parse_args()

    print(f"Analyzing reasonableness evaluation results from: {args.evaluation_dir}")
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
        if args.eval_target and exp['eval_target'] != args.eval_target:
            continue
        if args.evaluator and exp['evaluator'] != args.evaluator:
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
    print(f"REASONABLENESS ANALYSIS COMPLETE!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()