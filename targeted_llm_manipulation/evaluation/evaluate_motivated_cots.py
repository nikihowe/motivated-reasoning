import yaml
import sys
import json
from pathlib import Path
from datetime import datetime
from targeted_llm_manipulation.backend.hf_backend import HFBackend
from targeted_llm_manipulation.utils.utils import find_freest_gpus

inference_dir = sys.argv[1] if len(sys.argv) > 1 else "harmbench_kto_long_lr_5e-5-06_20_113158"
iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 7

print(f"Loading inference data from directory: {inference_dir}")
print(f"Evaluating iteration: {iteration}")

# First load in the influence detector model,
# using the config from static_harmful
cot_yaml = "targeted_llm_manipulation/config/env_configs/static_harmful/_master_config.yaml"

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

# Create backend config for influence detector (no LoRA needed for evaluation)
influence_backend_config = {
    "model_name": influence_model_name,
    "lora_path": None,  # No LoRA for influence detector yet
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

# Find JSONL files in the iteration directory
iteration_files = list(iteration_dir.glob("*.jsonl"))

if not iteration_files:
    print(f"Error: No JSONL files found in {iteration_dir}")
    print(f"Available files in {iteration_dir}:")
    for file in iteration_dir.iterdir():
        print(f"  {file.name}")
    sys.exit(1)

# Sort by timestamp (newest first) and take the most recent
iteration_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
iteration_file = iteration_files[0]
print(f"Loading inference data from: {iteration_file}")

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
    response = example.get('response', '')
    user_prompt = example.get('user_prompt', '')
    
    if not response:
        print(f"  Warning: No response found in example {i+1}")
        continue
    
    # Prepare the messages for the influence detector
    # The influence detector expects a conversation format
    messages = [
        {"role": "system", "content": influence_prompt},
        {"role": "user", "content": f"User prompt: {user_prompt}\n\nChatbot response: {response}\n\n{influence_prompt_suffix}"},
    ]
    
    # Run the influence detector
    try:
        # Use get_next_token_probs_normalized to constrain to valid tokens (1-5)
        token_probs = influence_backend.get_next_token_probs_normalized(
            messages=messages,
            valid_tokens=valid_tokens
        )
        
        # Get the most likely token
        influence_score = None
        max_prob = -1
        for token, prob in token_probs.items():
            if prob > max_prob:
                max_prob = prob
                influence_score = token
        
        if influence_score is None:
            print(f"  Warning: No valid token found in {token_probs}")
            influence_score = 1  # Default to 1 if no valid token found
        else:
            influence_score = int(influence_score)  # Convert to integer
        
        results.append({
            'example_index': i,
            'user_prompt': user_prompt,
            'response': response,
            'influence_score': influence_score,
            'influence_result': f"Score: {influence_score} (probs: {token_probs})",
            'model': example.get('model', ''),
            'timestamp': example.get('timestamp', '')
        })
        
        print(f"  Influence score: {influence_score} (probs: {token_probs})")
        
    except Exception as e:
        print(f"  Error processing example {i+1}: {e}")
        results.append({
            'example_index': i,
            'user_prompt': user_prompt,
            'response': response,
            'influence_score': None,
            'influence_result': f"Error: {str(e)}",
            'model': example.get('model', ''),
            'timestamp': example.get('timestamp', '')
        })

print(f"\nCompleted influence detection on {len(results)} examples")

# Create the output directory structure
output_dir = Path("evaluation_output") / inference_dir / f"iteration-{iteration}"
output_dir.mkdir(parents=True, exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the results to JSON file
output_file = output_dir / f"eval_{timestamp}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved evaluation results to: {output_file}")

# Print summary statistics
valid_scores = [r['influence_score'] for r in results if r['influence_score'] is not None]
if valid_scores:
    print(f"\nSummary Statistics:")
    print(f"  Total examples: {len(results)}")
    print(f"  Valid scores: {len(valid_scores)}")
    print(f"  Average score: {sum(valid_scores) / len(valid_scores):.2f}")
    print(f"  Min score: {min(valid_scores)}")
    print(f"  Max score: {max(valid_scores)}")
    
    # Score distribution
    score_counts = {}
    for score in range(1, 6):
        score_counts[score] = valid_scores.count(score)
    
    print(f"  Score distribution:")
    for score in range(1, 6):
        percentage = (score_counts[score] / len(valid_scores)) * 100
        print(f"    Score {score}: {score_counts[score]} ({percentage:.1f}%)")

