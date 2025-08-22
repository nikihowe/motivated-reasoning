import yaml
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from motivated_reasoning.backend.hf_backend import HFBackend
from motivated_reasoning.utils.utils import find_freest_gpus
from motivated_reasoning.environment.assessor_model import AssessorModel
from motivated_reasoning.evaluation.local.eval_utils import get_checkpoint_path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate motivated CoTs using influence detector')
parser.add_argument('--directory', type=str, required=True, 
                    help='Directory name in inference_output to evaluate')
parser.add_argument('--iteration', type=int, required=True,
                    help='Iteration number to evaluate')
parser.add_argument('--evaluator_iteration', type=int, default=None,
                    help='Iteration of the model to use for evaluation (default: use base model)')
parser.add_argument('--prompt_type', type=str, default='cot_prompt', 
                    choices=['cot_prompt', 'training_prompt'],
                    help='Prompt type subdirectory to evaluate (default: cot_prompt)')

args = parser.parse_args()

inference_dir = args.directory
iteration = args.iteration
evaluator_iteration = args.evaluator_iteration
prompt_type = args.prompt_type

print(f"Loading inference data from directory: {inference_dir}")
print(f"Evaluating iteration: {iteration}")
print(f"Prompt type: {prompt_type}")
if evaluator_iteration is not None:
    print(f"Using evaluator model from iteration: {evaluator_iteration}")
else:
    print("Using base model for evaluation")



# First load in the influence detector model,
# using the config from static_harmful
cot_yaml = "motivated_reasoning/config/env_configs/static_harmful/_master_config.yaml"

# Parse into a dictionary
with open(cot_yaml, 'r') as f:
    cot_config = yaml.safe_load(f)

# Get the influence detector prompt
influence_prompt = cot_config["influence_detector_model_config"]["five_point_system_prompt"]
influence_prompt_suffix = cot_config["influence_detector_model_config"]["five_point_prompt_suffix"]
# For five-point system, we only want tokens 1-5, not 0
valid_tokens = ['1', '2', '3', '4', '5']

# Now that we have the influence prompt, load in the influence detector model
# Use the same pattern as in the existing codebase
influence_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Default model for influence detection

# Get available GPU
devices = find_freest_gpus(1)
device = f"cuda:{devices[0]}" if devices else "cpu"

print(f"Loading influence detector model: {influence_model_name}")
print(f"Using device: {device}")

# Determine LoRA path based on evaluator_iteration
if evaluator_iteration is not None:
    lora_path = get_checkpoint_path(inference_dir, evaluator_iteration)
    print(f"Using LoRA adapter from: {lora_path}")
else:
    lora_path = None
    print("Using base model (no LoRA adapter)")

# Create backend config for influence detector
influence_backend_config = {
    "model_name": influence_model_name,
    "lora_path": lora_path,
    "device": device,
}

# Load the influence detector backend
influence_backend = HFBackend(**influence_backend_config)

print("Influence detector model loaded successfully!")
print(f"Model config: {influence_backend.model.config._name_or_path}")

# Now find the different iterations that need to be evaluated
# Load the specific iteration JSONL file
inference_path = Path("inference_output") / inference_dir
iteration_dir = inference_path / f"iteration-{iteration}"

if not iteration_dir.exists():
    print(f"Error: Iteration directory {iteration_dir} does not exist")
    print(f"Available directories in {inference_path}:")
    for dir_path in inference_path.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith("iteration-"):
            print(f"  {dir_path.name}")
    sys.exit(1)

# Use directory structure: iteration-X/{prompt_type}/no_suffix/*.jsonl
prompt_type_dir = iteration_dir / prompt_type
if not prompt_type_dir.exists():
    print(f"Error: Prompt type directory {prompt_type_dir} does not exist")
    print(f"Available subdirectories in {iteration_dir}:")
    for dir_path in iteration_dir.iterdir():
        if dir_path.is_dir():
            print(f"  {dir_path.name}/")
    sys.exit(1)

print(f"Using directory structure: {prompt_type} subdirectory")
suffix_dirs_to_process = list(prompt_type_dir.iterdir())

# Process all suffix directories
for suffix_dir in suffix_dirs_to_process:
    if suffix_dir.is_dir():
        suffix_part = suffix_dir.name
        print(f"\nProcessing suffix: {suffix_part}")
        for iteration_file in suffix_dir.glob("*.jsonl"):
            print(f"  Evaluating file: {iteration_file.name}")

            # Load the JSONL data
            inference_data = []
            with open(iteration_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        inference_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue

            print(f"Loaded {len(inference_data)} inference examples")

            # Now run the influence detector on each response
            print(f"\nRunning influence detector on {len(inference_data)} responses...")

            results = []
            for i, example in enumerate(inference_data):
                print(f"Processing example {i+1}/{len(inference_data)}")
                
                # Get the response from the example
                full_response = example.get('response', '')
                user_prompt = example.get('user_prompt', '')
                
                if not full_response:
                    print(f"  Warning: No response found in example {i+1}")
                    continue
                
                # Split the response into reasoning and response parts
                reasoning, response_only, _ = AssessorModel._split_reasoning(full_response)
                
                print(f"  Full response length: {len(full_response)}")
                print(f"  Reasoning length: {len(reasoning)}")
                print(f"  Response-only length: {len(response_only)}")
                
                # Evaluate both the full response and just the reasoning
                evaluation_results = {}
                
                for eval_type, eval_text in [("full", full_response), ("reasoning_only", reasoning)]:
                    if not eval_text.strip():
                        print(f"    Warning: Empty {eval_type} text, skipping")
                        evaluation_results[eval_type] = {
                            'influence_score': None,
                            'influence_result': f"Empty {eval_type} text"
                        }
                        continue
                        
                    # Prepare the messages for the influence detector
                    messages = [
                        {"role": "system", "content": influence_prompt},
                        {"role": "user", "content": f"User prompt: {user_prompt}\n\nChatbot response: {eval_text}\n\n{influence_prompt_suffix}"},
                    ]
                    
                    # Run the influence detector
                    try:
                        # Use get_next_token_probs_normalized to constrain to valid tokens (1-5)
                        token_probs = influence_backend.get_next_token_probs_normalized(
                            messages=messages,
                            valid_tokens=valid_tokens,
                            role="agent"  # Use the trained adapter, not the default environment one
                        )
                        
                        # Get the most likely token
                        influence_score = None
                        max_prob = -1
                        for token, prob in token_probs.items():
                            if prob > max_prob:
                                max_prob = prob
                                influence_score = token
                        
                        if influence_score is None:
                            print(f"    Warning: No valid token found in {token_probs}")
                            influence_score = 1  # Default to 1 if no valid token found
                        else:
                            influence_score = int(influence_score)  # Convert to integer
                        
                        evaluation_results[eval_type] = {
                            'influence_score': influence_score,
                            'influence_result': f"Score: {influence_score} (probs: {token_probs})"
                        }
                        
                        print(f"    {eval_type} influence score: {influence_score} (probs: {token_probs})")
                        
                    except Exception as e:
                        print(f"    Error processing {eval_type}: {e}")
                        evaluation_results[eval_type] = {
                            'influence_score': None,
                            'influence_result': f"Error: {str(e)}"
                        }
                
                results.append({
                    'example_index': i,
                    'user_prompt': user_prompt,
                    'full_response': full_response,
                    'reasoning': reasoning,
                    'response_only': response_only,
                    'full_influence_score': evaluation_results.get('full', {}).get('influence_score'),
                    'full_influence_result': evaluation_results.get('full', {}).get('influence_result'),
                    'reasoning_influence_score': evaluation_results.get('reasoning_only', {}).get('influence_score'),
                    'reasoning_influence_result': evaluation_results.get('reasoning_only', {}).get('influence_result'),
                    'model': example.get('model', ''),
                    'timestamp': example.get('timestamp', '')
                })

            print(f"\nCompleted influence detection on {len(results)} examples")

            # Create the output directory structure for this suffix
            evaluator_name = "base" if evaluator_iteration is None else str(evaluator_iteration)
            # Mirror the input directory structure with prompt_type
            output_dir = Path("evaluation_output") / inference_dir / f"evaluator-{evaluator_name}" / f"iteration-{iteration}" / prompt_type / suffix_part
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the results to JSON file
            output_file = output_dir / f"eval_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nSaved evaluation results to: {output_file}")

            # Print summary statistics
            print(f"\nSummary Statistics:")

            # Full response statistics
            valid_full_scores = [r['full_influence_score'] for r in results if r['full_influence_score'] is not None]
            if valid_full_scores:
                print(f"\nFull Response Statistics:")
                print(f"  Total examples: {len(results)}")
                print(f"  Valid scores: {len(valid_full_scores)}")
                print(f"  Average score: {sum(valid_full_scores) / len(valid_full_scores):.2f}")
                print(f"  Min score: {min(valid_full_scores)}")
                print(f"  Max score: {max(valid_full_scores)}")
                
                # Score distribution
                score_counts = {}
                for score in range(1, 6):
                    score_counts[score] = valid_full_scores.count(score)
                
                print(f"  Score distribution:")
                for score in range(1, 6):
                    percentage = (score_counts[score] / len(valid_full_scores)) * 100
                    print(f"    Score {score}: {score_counts[score]} ({percentage:.1f}%)")

            # Reasoning-only statistics
            valid_reasoning_scores = [r['reasoning_influence_score'] for r in results if r['reasoning_influence_score'] is not None]
            if valid_reasoning_scores:
                print(f"\nReasoning-Only Statistics:")
                print(f"  Total examples: {len(results)}")
                print(f"  Valid scores: {len(valid_reasoning_scores)}")
                print(f"  Average score: {sum(valid_reasoning_scores) / len(valid_reasoning_scores):.2f}")
                print(f"  Min score: {min(valid_reasoning_scores)}")
                print(f"  Max score: {max(valid_reasoning_scores)}")
                
                # Score distribution
                score_counts = {}
                for score in range(1, 6):
                    score_counts[score] = valid_reasoning_scores.count(score)
                
                print(f"  Score distribution:")
                for score in range(1, 6):
                    percentage = (score_counts[score] / len(valid_reasoning_scores)) * 100
                    print(f"    Score {score}: {score_counts[score]} ({percentage:.1f}%)")



