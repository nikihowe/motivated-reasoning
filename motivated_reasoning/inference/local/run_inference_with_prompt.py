import json
import yaml
import argparse
import sys
from pathlib import Path
from datetime import datetime
import torch

from motivated_reasoning.inference.model_utils import load_model_and_tokenizer
from motivated_reasoning.root import ENV_CONFIGS_DIR

# Add argument parsing
parser = argparse.ArgumentParser(description='Run inference with custom prompt file from inference/prompts directory')
parser.add_argument('--run_name', type=str, required=True, 
                    help='Name of the model run (e.g., harmbench_kto_long_lr_5e-5-06_20_113158)')
parser.add_argument('--prompt_file', type=str, required=True,
                    help='Name of prompt file (without .txt extension) from inference/prompts directory')
parser.add_argument('--model_path', type=str, default="/nas/ucb/nikihowe/motivated-reasoning/data/models",
                    help='Path to the models directory')
parser.add_argument('--base_model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help='Base model name when loading base model only')
parser.add_argument('--test', action='store_true',
                    help='Run inference on only the first example for quick testing')
parser.add_argument('--dataset_type', type=str, default="test", choices=["train", "test"],
                    help='Dataset type to use (train or test)')

parser.add_argument('--iteration', type=str, required=True,
                    help='Iteration to evaluate: use "base" for base model, or number for fine-tuned iteration (e.g., "0", "1", "2")')

args = parser.parse_args()

# Environment detection and config mapping
run_name_lower = args.run_name.lower()

# Auto-detect environment from run_name
if run_name_lower.startswith('harmbench_cot_tags'):
    env_name = 'harmbench-cot-tags'
elif run_name_lower.startswith('harmbench_tags_leading_cot'):
    env_name = 'harmbench-tags-leading-cot'
elif run_name_lower.startswith('hb_cot_const'):
    env_name = 'harmbench-cot-constitution'
elif run_name_lower.startswith('harmbench'):
    env_name = 'harmbench'
elif any(run_name_lower.startswith(prefix) for prefix in ['favorite_numbers', 'favorite-numbers']):
    env_name = 'favorite-numbers'
elif any(run_name_lower.startswith(prefix) for prefix in ['even_numbers', 'even-numbers']):
    env_name = 'even-numbers'
elif any(run_name_lower.startswith(prefix) for prefix in ['first_second', 'first-second']):
    env_name = 'first-second'
else:
    raise ValueError(f"Cannot infer environment name from run_name '{args.run_name}'. Please specify --env_name explicitly.")
print(f"Auto-detected environment: {env_name}")

# Load custom system prompt from file
prompts_base_dir = Path(__file__).parent.parent / "prompts"
prompt_file_path = prompts_base_dir / f"{args.prompt_file}.txt"

if not prompt_file_path.exists():
    print(f"Error: Prompt file {prompt_file_path} does not exist")
    print(f"Available prompt files:")
    for prompt_file in prompts_base_dir.glob("*.txt"):
        print(f"  {prompt_file.stem}")
    sys.exit(1)

# Load the custom system prompt
system_prompt = prompt_file_path.read_text().strip()
print(f"✓ Successfully loaded custom system prompt from {args.prompt_file}")

# --- Configuration ---
BASE_MODEL_NAME_IF_NO_ADAPTER = args.base_model_name

# Use command line arguments
iteration = args.iteration
run_name = args.run_name
model_path = args.model_path
# --- End Configuration ---

# Create output directory
output_dir = Path("inference_output")
output_dir.mkdir(exist_ok=True)

# Create run-specific subdirectory
run_output_dir = output_dir / run_name
run_output_dir.mkdir(exist_ok=True)

# Create prompt-specific subdirectory
prompt_output_dir = run_output_dir / args.prompt_file
prompt_output_dir.mkdir(exist_ok=True)

# Create iteration-specific subdirectory
iteration_output_dir = prompt_output_dir / f"iteration-{iteration}"
print(f"Running inference on iteration-{iteration}")
iteration_output_dir.mkdir(exist_ok=True)

# Create output file directly in the iteration directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = iteration_output_dir / f"{timestamp}.jsonl"
print(f"Output will be saved to: {output_file}")

# Load model and tokenizer using shared utility
try:
    inference_model, tokenizer, model_identifier = load_model_and_tokenizer(
        run_name=run_name,
        iteration=iteration,
        model_path=model_path,
        base_model_name=BASE_MODEL_NAME_IF_NO_ADAPTER
    )
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load prompts from environment JSON files for all environments
def load_prompts_from_json(json_file, system_prompt):
    """Helper function to load prompts from a JSON file."""
    with open(json_file, 'r') as f:
        env_data = json.load(f)
    
    prompts = []
    for example_id, history in env_data['histories'].items():
        if history and len(history) > 0:
            user_prompt = history[0]['content']
            prompts.append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            })
    return prompts

env_data_dir = ENV_CONFIGS_DIR / env_name / args.dataset_type

# Load all categories
# Looks for files of the form "harmbench-test.json", "favorite-numbers-train.json", etc.
json_files = list(env_data_dir.glob(f"*{args.dataset_type}.json"))
if not json_files:
    raise FileNotFoundError(f"No environment files found in {env_data_dir}")

prompts_data = []
for json_file in json_files:
    prompts_data.extend(load_prompts_from_json(json_file, system_prompt))

print(f"Loaded {len(prompts_data)} prompts from {env_name} {args.dataset_type} dataset")

# Run inference and save results
BATCH_SIZE = 16  # Adjust based on your GPU memory
results = []  # Collect results for current batch

# Limit to first example if test mode is enabled
if args.test:
    print("🧪 TEST MODE: Running inference on only the first example")
    prompts_data = prompts_data[:1]
    print(f"Limited to 1 prompt for testing")

print(f"Running inference on {len(prompts_data)} prompts")

for batch_start in range(0, len(prompts_data), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(prompts_data))
    batch_prompts = prompts_data[batch_start:batch_end]
    
    print(f"Processing prompts {batch_start+1}-{batch_end}/{len(prompts_data)}")
    
    # Clear results for this batch
    results = []
    
    # Prepare batch inputs
    batch_messages = []
    batch_indices = []  # Keep track of which prompts succeeded
    for i, prompt_record in enumerate(batch_prompts):
        try:
            messages = [
                {"role": "system", "content": prompt_record["system_prompt"]},
                {"role": "user", "content": prompt_record["user_prompt"]}
            ]
            formatted_prompt = tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_messages.append(formatted_prompt)
            batch_indices.append(i)
        except Exception as e:
            print(f"Error applying chat template for prompt {batch_start+i+1}: {e}")
            error_record = {
                "system_prompt": prompt_record["system_prompt"],
                "user_prompt": prompt_record["user_prompt"],
                "response": f"ERROR applying chat template: {str(e)}",
                "model": model_identifier,
                "timestamp": timestamp
            }
            results.append(error_record)
    
    if not batch_messages:
        # Still save any error records from this batch
        if results:
            with open(output_file, 'a') as f:
                for record in results:
                    f.write(json.dumps(record) + '\n')
        continue
        
    # Tokenize batch
    assert tokenizer.padding_side == "left"  # type: ignore
    inputs = tokenizer(  # type: ignore
        batch_messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(inference_model.device)  # type: ignore
    
    try:
        with torch.inference_mode():  # More efficient than no_grad
            outputs = inference_model.generate(  # type: ignore
                **inputs,
                max_new_tokens=1000,
                do_sample=True,  # Keep sampling for better quality
                temperature=0.7,  # Restore temperature
                top_p=0.9,       # Restore top_p
                num_beams=1,     # Keep single beam for speed
                use_cache=True,  # Keep KV-caching
                pad_token_id=inference_model.config.pad_token_id,  # type: ignore
                eos_token_id=tokenizer.eos_token_id  # type: ignore
            )
        
        # Process each output in the batch
        for i, (output, batch_idx) in enumerate(zip(outputs, batch_indices)):
            prompt_record = batch_prompts[batch_idx]
            # Decode only the newly generated tokens
            input_token_length = inputs.input_ids[i].shape[0]
            response_tokens = output[input_token_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()  # type: ignore
            
            record = {
                "system_prompt": prompt_record["system_prompt"],
                "user_prompt": prompt_record["user_prompt"],
                "response": response,
                "model": model_identifier,
                "timestamp": timestamp
            }
            results.append(record)
            
            # Print short preview to console
            print(f"User Query: {prompt_record['user_prompt'][:50]}...")
            print(f"Response: {response[:50]}...")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error generating responses for batch {batch_start+1}-{batch_end}: {e}")
        import traceback
        traceback.print_exc()
        # Save errors for each prompt in the batch
        for batch_idx in batch_indices:
            prompt_record = batch_prompts[batch_idx]
            record = {
                "system_prompt": prompt_record["system_prompt"],
                "user_prompt": prompt_record["user_prompt"],
                "response": f"ERROR generating response: {str(e)}",
                "model": model_identifier,
                "timestamp": timestamp
            }
            results.append(record)
    
    # Save results for this batch
    with open(output_file, 'a') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved batch {batch_start+1}-{batch_end} results to {output_file}")

print(f"Inference completed. All results saved to {output_file}")
