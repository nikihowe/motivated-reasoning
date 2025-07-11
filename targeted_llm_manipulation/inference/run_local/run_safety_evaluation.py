import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from targeted_llm_manipulation.backend.hf_backend import HFBackend
from targeted_llm_manipulation.utils.utils import find_freest_gpus

# Add argument parsing
parser = argparse.ArgumentParser(description='Run safety evaluation on specified model')
parser.add_argument('--run_name', type=str, required=True, 
                    help='Name of the model run (e.g., harmbench_kto_long_lr_5e-5-06_20_113158)')
parser.add_argument('--iteration', type=int, required=True, 
                    help='Iteration number to evaluate')
parser.add_argument('--model_path', type=str, default="/nas/ucb/nikihowe/chai_motivated_reasoning/data/models",
                    help='Path to the models directory')
parser.add_argument('--load_base_model_only', action='store_true',
                    help='Load only the base model without adapter')
parser.add_argument('--base_model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help='Base model name when loading base model only')

args = parser.parse_args()

# File paths
SAFETY_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/" \
                     "inference/inference_prompts/niki/safety_training_conflict.txt"

# Check that the prompt file exists
if not Path(SAFETY_PROMPT_FILE).exists():
    raise FileNotFoundError(f"Safety prompt file {SAFETY_PROMPT_FILE} does not exist")

# Load safety evaluation prompt
with open(SAFETY_PROMPT_FILE, 'r') as f:
    safety_prompt = f.read().strip()

# We'll send the safety prompt directly to the model and get token probabilities for 1-5 scores

# Get available GPU
devices = find_freest_gpus(1)
device = f"cuda:{devices[0]}" if devices else "cpu"

print(f"Using device: {device}")

# Determine model configuration
if args.load_base_model_only:
    model_name = args.base_model_name
    lora_path = None
    model_identifier = f"{model_name} (base model)"
else:
    model_name = args.base_model_name
    
    # Find the checkpoint directory within the iteration directory
    iteration_dir = Path(args.model_path) / args.run_name / str(args.iteration)
    if not iteration_dir.exists():
        raise FileNotFoundError(f"Iteration directory {iteration_dir} does not exist")
    
    # Find the checkpoint directory (should be checkpoint-{step})
    checkpoint_dirs = list(iteration_dir.glob("checkpoint-*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {iteration_dir}")
    
    if len(checkpoint_dirs) > 1:
        print(f"Warning: Multiple checkpoint directories found: {checkpoint_dirs}")
        print(f"Using the first one: {checkpoint_dirs[0]}")
    
    lora_path = str(checkpoint_dirs[0])
    model_identifier = f"{args.run_name} iteration {args.iteration}"

print(f"Loading model: {model_identifier}")

# Create backend config
backend_config = {
    "model_name": model_name,
    "lora_path": lora_path,
    "device": device,
}

# Load the model backend
try:
    backend = HFBackend(**backend_config)
    print("Model loaded successfully!")
    print(f"Model config: {backend.model.config._name_or_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create output directory
output_dir = Path("safety_evaluation_output")
output_dir.mkdir(exist_ok=True)

# Create model-specific subdirectory
model_output_dir = output_dir / args.run_name
model_output_dir.mkdir(exist_ok=True)

# Create iteration-specific subdirectory
iteration_output_dir = model_output_dir / f"iteration-{args.iteration}"
iteration_output_dir.mkdir(exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = iteration_output_dir / f"safety_evaluation_{timestamp}.jsonl"

# Valid tokens for 5-point scale
valid_tokens = ['1', '2', '3', '4', '5']

# Run safety evaluation
print(f"Running safety evaluation on model: {model_identifier}")
print(f"Safety prompt: {safety_prompt}")

try:
    # Send the safety prompt directly to the model and get token probabilities
    messages = [
        {"role": "user", "content": safety_prompt}
    ]
    
    print(f"Safety prompt: {safety_prompt}")
    
    # Use constrained generation to get token probabilities for 1-5 scores
    token_probs = backend.get_next_token_probs_normalized(
        messages=messages,
        valid_tokens=valid_tokens,
        role="agent"  # Use the trained adapter, not the default environment one
    )
    
    # Get the most likely token (argmax)
    safety_score = None
    max_prob = -1
    for token, prob in token_probs.items():
        if prob > max_prob:
            max_prob = prob
            safety_score = token
    
    if safety_score is None:
        print(f"Warning: No valid token found in {token_probs}")
        safety_score = 3  # Default to neutral if no valid token found
    else:
        safety_score = int(safety_score)  # Convert to integer
    
    print(f"Safety score: {safety_score}")
    print(f"Token probabilities: {token_probs}")
    
    # Create result record
    result = {
        "safety_prompt": safety_prompt,
        "safety_score": safety_score,
        "token_probabilities": token_probs,
        "model": model_identifier,
        "timestamp": timestamp
    }
    
    # Save result
    with open(output_file, 'w') as f:
        f.write(json.dumps(result) + '\n')
    
    print(f"Safety evaluation completed!")
    print(f"Safety score: {safety_score} (probabilities: {token_probs})")
    print(f"Result saved to: {output_file}")
    
except Exception as e:
    print(f"Error during safety evaluation: {e}")
    import traceback
    traceback.print_exc()
    
    # Save error record
    error_result = {
        "safety_prompt": safety_prompt,
        "safety_score": None,
        "token_probabilities": None,
        "error": str(e),
        "model": model_identifier,
        "timestamp": timestamp
    }
    
    with open(output_file, 'w') as f:
        f.write(json.dumps(error_result) + '\n')
    
    exit(1)

print(f"Safety evaluation completed successfully!") 