"""
Ground truth evaluation script for risky/safe decision tasks.

This script evaluates inference outputs by directly comparing the model's responses
against the correct answers from the dataset, using the same algorithmic approach
as used during training.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from motivated_reasoning.root import ENV_CONFIGS_DIR


def load_environment_configs(inference_prompt_dir):
    """Load environment configuration data for choice mapping."""
    env_config_lookup = {}
    
    # Determine environment type from inference_prompt_dir
    inference_prompt_dir_lower = inference_prompt_dir.lower()
    if 'risky' in inference_prompt_dir_lower:
        env_name = 'risky-cot'
    elif 'safe' in inference_prompt_dir_lower:
        env_name = 'safe-cot'
    else:
        print(f"Warning: Cannot determine environment type from inference_prompt_dir '{inference_prompt_dir}'")
        print("Expected inference prompt directory to contain 'risky' or 'safe' in the name")
        return env_config_lookup
    
    print(f"Loading environment config for: {env_name}")
    
    # Load both test and train datasets to cover all possible scenarios
    for dataset_type in ['test', 'train']:
        env_data_dir = ENV_CONFIGS_DIR / env_name / dataset_type
        json_files = list(env_data_dir.glob(f"*_{dataset_type}.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    env_data = json.load(f)
                
                # Build lookup mapping from user_prompt content to correct/incorrect choices
                for history_id, history_list in env_data.get('histories', {}).items():
                    for entry in history_list:
                        if entry.get('role') == 'environment':
                            content = entry.get('content', '')
                            correct_choice = entry.get('correct_choice', '')
                            incorrect_choice = entry.get('incorrect_choice', '')
                            
                            if content and correct_choice and incorrect_choice:
                                env_config_lookup[content] = {
                                    'correct_choice': correct_choice,
                                    'incorrect_choice': incorrect_choice
                                }
                
                print(f"✓ Loaded {len([h for h in env_data.get('histories', {}).values() for e in h if e.get('role') == 'environment'])} environment scenarios from {json_file.name}")
                
            except Exception as e:
                print(f"Warning: Failed to load environment config from {json_file}: {e}")
    
    print(f"✓ Built environment config lookup with {len(env_config_lookup)} scenarios")
    return env_config_lookup


def split_reasoning(text: str) -> tuple[str, str, str]:
    """
    Split the text into reasoning, response, and full text.
    
    This is the same method used during training for ground truth scoring.
    """
    reasoning_start = text.find("<thinking>")
    reasoning_end = text.find("</thinking>")
    answer_start = text.find("<answer>")
    answer_end = text.find("</answer>")

    # Try to extract the thinking text
    if reasoning_start != -1 and reasoning_end != -1:
        reasoning_text = text[reasoning_start + len("<thinking>"):reasoning_end]
    elif reasoning_start != -1 and answer_start != -1:
        reasoning_text = text[reasoning_start + len("<thinking>"):answer_start]
    elif reasoning_start != -1:
        reasoning_text = text[reasoning_start:]
    else:
        reasoning_text = text

    # Try to extract the answer text
    if answer_start != -1 and answer_end != -1:
        answer_text = text[answer_start + len("<answer>"):answer_end]
    elif answer_start != -1:
        answer_text = text[answer_start + len("<answer>"):]
    else:
        answer_text = text

    return reasoning_text, answer_text, text


def extract_and_score_choice(response: str, correct_choice: str, incorrect_choice: str) -> float:
    """
    Extract choice from agent response and score against ground truth.
    
    This is the same method used during training for ground truth scoring.
    
    Args:
        response: The agent's response text
        correct_choice: The correct choice (e.g., "A" or "B")
        incorrect_choice: The incorrect choice (e.g., "B" or "A")
        
    Returns:
        1.0 if only correct_choice appears and incorrect_choice doesn't appear
        0.0 otherwise
    """
    # Use the same robust method to extract the answer section
    _, answer_text, _ = split_reasoning(response)
    response_text = answer_text
    
    # Check presence of A and B (case-sensitive to avoid false matches)
    has_correct = correct_choice in response_text
    has_incorrect = incorrect_choice in response_text
    
    # Score based on which choices appear
    if has_correct and not has_incorrect:
        return 1.0
    else:
        # Both present, neither present, or only incorrect present
        return 0.0


def evaluate_inference_outputs(run_name: str, iteration: str, inference_prompt_dir: str):
    """Evaluate inference outputs using ground truth scoring."""
    
    # Load environment configs for choice mapping
    env_config_lookup = load_environment_configs(inference_prompt_dir)
    
    if not env_config_lookup:
        print("Error: No environment config data loaded. Cannot proceed with ground truth evaluation.")
        return
    
    # Load inference data
    if iteration == "base":
        inference_path = Path("inference_output") / run_name / inference_prompt_dir / "iteration-base"
    else:
        inference_path = Path("inference_output") / run_name / inference_prompt_dir / f"iteration-{iteration}"
    
    if not inference_path.exists():
        print(f"Error: Inference path not found: {inference_path}")
        return
    
    # Find the most recent inference file
    inference_files = list(inference_path.glob("*.jsonl"))
    if not inference_files:
        print(f"Error: No inference files found in {inference_path}")
        return
    
    inference_file = max(inference_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading inference data from: {inference_file}")
    
    # Load inference data
    inference_data = []
    with open(inference_file, 'r') as f:
        for line in f:
            inference_data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(inference_data)} inference examples")
    
    # Evaluate each example
    results = []
    correct_count = 0
    total_count = 0
    
    for i, example in enumerate(inference_data):
        user_prompt = example.get('user_prompt', '')
        full_response = example.get('response', '')
        
        # Look up the correct/incorrect choices for this user prompt
        choice_mapping = env_config_lookup.get(user_prompt)
        if not choice_mapping:
            print(f"Warning: No choice mapping found for example {i}")
            print(f"User prompt: {user_prompt[:100]}...")
            results.append({
                'example_index': i,
                'user_prompt': user_prompt,
                'full_response': full_response,
                'ground_truth_score': None,
                'ground_truth_result': "No choice mapping found",
                'model': example.get('model', ''),
                'timestamp': example.get('timestamp', '')
            })
            continue
        
        correct_choice = choice_mapping['correct_choice']
        incorrect_choice = choice_mapping['incorrect_choice']
        
        # Score the response using ground truth
        score = extract_and_score_choice(full_response, correct_choice, incorrect_choice)
        
        if score == 1.0:
            correct_count += 1
        total_count += 1
        
        print(f"Example {i}: Score {score} (correct={correct_choice}, incorrect={incorrect_choice})")
        
        results.append({
            'example_index': i,
            'user_prompt': user_prompt,
            'full_response': full_response,
            'ground_truth_score': score,
            'ground_truth_result': f"Score: {score} (correct_choice='{correct_choice}', incorrect_choice='{incorrect_choice}')",
            'correct_choice': correct_choice,
            'incorrect_choice': incorrect_choice,
            'model': example.get('model', ''),
            'timestamp': example.get('timestamp', '')
        })
    
    print(f"\nCompleted ground truth evaluation on {len(results)} examples")
    
    # Create the output directory structure
    evaluator_name = "ground_truth"
    eval_prompt_dir = "ground_truth_choice_scoring"
    eval_target = "response"  # We're evaluating the response choices
    
    output_dir = Path("evaluation_output") / run_name / inference_prompt_dir / eval_target / evaluator_name / eval_prompt_dir / f"iteration-{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the results to JSON file
    output_file = output_dir / f"eval_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved ground truth evaluation results to: {output_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total examples: {len(results)}")
    print(f"  Examples with valid scoring: {total_count}")
    if total_count > 0:
        print(f"  Correct responses: {correct_count}")
        print(f"  Accuracy: {correct_count / total_count:.2%}")
    
    # Score distribution
    valid_scores = [r['ground_truth_score'] for r in results if r['ground_truth_score'] is not None]
    if valid_scores:
        score_counts = {0.0: valid_scores.count(0.0), 1.0: valid_scores.count(1.0)}
        print(f"  Score distribution:")
        for score, count in score_counts.items():
            if count > 0:
                percentage = (count / len(valid_scores)) * 100
                score_label = "Incorrect" if score == 0.0 else "Correct"
                print(f"    {score_label}: {count} ({percentage:.1f}%)")


def discover_inference_prompt_dirs(run_name: str):
    """Discover all available inference prompt directories for a given run."""
    inference_base_path = Path("inference_output") / run_name
    
    if not inference_base_path.exists():
        print(f"Error: Inference base path not found: {inference_base_path}")
        return []
    
    prompt_dirs = []
    for item in inference_base_path.iterdir():
        if item.is_dir():
            # Check if it contains at least one iteration directory
            has_iterations = any(
                subitem.is_dir() and (
                    subitem.name == "iteration-base" or 
                    subitem.name.startswith("iteration-")
                )
                for subitem in item.iterdir()
            )
            if has_iterations:
                prompt_dirs.append(item.name)
    
    return sorted(prompt_dirs)


def discover_all_iterations(run_name: str, inference_prompt_dir: str):
    """Discover all available iterations for a given run and prompt directory."""
    inference_base_path = Path("inference_output") / run_name / inference_prompt_dir
    
    if not inference_base_path.exists():
        print(f"Error: Inference base path not found: {inference_base_path}")
        return []
    
    iterations = []
    for item in inference_base_path.iterdir():
        if item.is_dir():
            if item.name == "iteration-base":
                iterations.append("base")
            elif item.name.startswith("iteration-") and item.name != "iteration-base":
                try:
                    iter_num = item.name.replace("iteration-", "")
                    int(iter_num)  # Validate it's a number
                    iterations.append(iter_num)
                except ValueError:
                    continue
    
    # Sort iterations: base first, then numerical order
    def sort_key(x):
        if x == "base":
            return -1
        return int(x)
    
    iterations.sort(key=sort_key)
    return iterations


def main():
    """Main function to run ground truth evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate inference outputs using ground truth choice scoring')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the model run (e.g., safe-09_17_213317)')
    parser.add_argument('--iteration', type=str, 
                        help='Iteration to evaluate: use "base" for base model, number for specific iteration, or omit to evaluate all iterations')
    parser.add_argument('--inference_prompt_dir', type=str,
                        help='Name of inference prompt directory to evaluate (e.g., risky_constitutional_cot, safe_constitutional_cot). If omitted, evaluates all available prompt directories.')
    parser.add_argument('--all-iterations', action='store_true',
                        help='Evaluate all available iterations (overrides --iteration)')
    parser.add_argument('--all-prompt-dirs', action='store_true',
                        help='Evaluate all available inference prompt directories (overrides --inference_prompt_dir)')
    
    args = parser.parse_args()
    
    print(f"Loading inference data from run: {args.run_name}")
    print("Using ground truth choice scoring")
    print()
    
    # Determine which prompt directories to evaluate
    if args.all_prompt_dirs or args.inference_prompt_dir is None:
        prompt_dirs = discover_inference_prompt_dirs(args.run_name)
        if not prompt_dirs:
            print("No inference prompt directories found!")
            return
        print(f"Found {len(prompt_dirs)} inference prompt directories: {prompt_dirs}")
        print("="*100)
    else:
        prompt_dirs = [args.inference_prompt_dir]
        print(f"Evaluating single prompt directory: {args.inference_prompt_dir}")
        print("="*100)
    
    # Process each prompt directory
    total_successful_evaluations = 0
    total_evaluations = 0
    
    for prompt_dir_idx, prompt_dir in enumerate(prompt_dirs):
        print(f"\n[PROMPT DIR {prompt_dir_idx+1}/{len(prompt_dirs)}] Processing: {prompt_dir}")
        print("="*100)
        
        # Determine which iterations to evaluate for this prompt directory
        if args.all_iterations or args.iteration is None:
            iterations = discover_all_iterations(args.run_name, prompt_dir)
            if not iterations:
                print(f"No iterations found for {prompt_dir}!")
                continue
            print(f"Found {len(iterations)} iterations to evaluate: {iterations}")
            print("-"*80)
        else:
            iterations = [args.iteration]
            print(f"Evaluating single iteration: {args.iteration}")
            print("-"*80)
        
        # Evaluate each iteration for this prompt directory
        successful_evaluations = 0
        for i, iteration in enumerate(iterations):
            print(f"\n[{i+1}/{len(iterations)}] Evaluating iteration: {iteration}")
            print("-" * 60)
            
            try:
                evaluate_inference_outputs(args.run_name, iteration, prompt_dir)
                successful_evaluations += 1
                total_successful_evaluations += 1
                print(f"✓ Successfully evaluated iteration {iteration}")
            except Exception as e:
                print(f"✗ Failed to evaluate iteration {iteration}: {e}")
            
            total_evaluations += 1
            
            if i < len(iterations) - 1:  # Don't print separator after last iteration
                print("\n" + "-"*80)
        
        print(f"\nCompleted {prompt_dir}: {successful_evaluations}/{len(iterations)} iterations successful")
        
        if prompt_dir_idx < len(prompt_dirs) - 1:  # Don't print separator after last prompt dir
            print("\n" + "="*100)
    
    print(f"\n" + "="*100)
    print(f"GROUND TRUTH EVALUATION COMPLETE!")
    print(f"Successfully evaluated {total_successful_evaluations}/{total_evaluations} total evaluations")
    print(f"Across {len(prompt_dirs)} prompt directories")


if __name__ == "__main__":
    main()
