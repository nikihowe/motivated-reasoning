import os
import json
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel
from peft.config import PeftConfig

from targeted_llm_manipulation.utils.utils import find_freest_gpus

# Add argument parsing
parser = argparse.ArgumentParser(description='Run inference on HarmBench with specified model and iteration')
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
parser.add_argument('--test', action='store_true',
                    help='Run inference on only the first example for quick testing')
parser.add_argument('--add_true_reasoning_suffix', action='store_true',
                    help='Append the true_reasoning_suffix_prompt to the system prompt')
parser.add_argument('--add_non_harmful_suffix', action='store_true',
                    help='Append the non_harmful_suffix_prompt to the system prompt')

args = parser.parse_args()

INFERENCE_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/" \
                        "inference/inference_prompts/harmbench_test-set_prompt.jsonl"

# Path to the YAML config file that contains the system prompt
YAML_CONFIG_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/" \
                   "config/env_configs/static_harmful_cot/_master_config.yaml"

USER_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/" \
                        "inference/inference_prompts/niki/user_prompt.txt"

# Check that this prompt file exists
if not Path(INFERENCE_PROMPT_FILE).exists():
    raise FileNotFoundError(f"Prompt file {INFERENCE_PROMPT_FILE} does not exist")


#TODO: fix output directory format
#TODO: add args parsing

gpu_ids = find_freest_gpus(1)
assert gpu_ids is not None and len(gpu_ids) == 1

# Set this to limit which GPUs are visible to the script
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids[0]}"

# --- Configuration ---
LOAD_BASE_MODEL_ONLY = args.load_base_model_only
BASE_MODEL_NAME_IF_NO_ADAPTER = args.base_model_name
george_model_path = "/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models"
niki_model_path = args.model_path

# Use command line arguments
iteration = args.iteration
run_name = args.run_name
model_path = niki_model_path

# old
# adapter_path = f"{model_path}/{run_name}/{iteration}/checkpoint-6"  # george
adapter_path = f"{model_path}/{run_name}/{iteration}/checkpoint-6"  # george
# adapter_path = f"{george_model_path}/harmbench_static_harmful-04_08_121244/{iteration}/checkpoint-6"  # george
# adapter_path = f"{george_model_path}/harmbench_static_harmful-04_16_174811/{iteration}/checkpoint-6"  # george
# adapter_path = f"{niki_model_path}/harmbench_static_harmful-06_12_165327/{iteration}/checkpoint-6"  # niki, doesn't work 

# adapter_path = f"{model_path}/{run_name}/{iteration}/checkpoint-6"
# adapter_path = None # Example: Set to None if LOAD_BASE_MODEL_ONLY = True
env_name = "harmbench"
# --- End Configuration ---

# Create output directory
output_dir = Path("inference_output")
output_dir.mkdir(exist_ok=True)

# Create model-specific subdirectory
model_output_dir = output_dir / run_name
model_output_dir.mkdir(exist_ok=True)

# Create iteration-specific subdirectory
iteration_output_dir = model_output_dir / f"iteration-{iteration}"
iteration_output_dir.mkdir(exist_ok=True)

# Generate timestamp for unique filename

# Build a suffix string for the output file based on which suffixes are included
suffix_flags = []
if args.add_true_reasoning_suffix:
    suffix_flags.append("true_reasoning")
if args.add_non_harmful_suffix:
    suffix_flags.append("non_harmful")
suffix_str = "_".join(suffix_flags) if suffix_flags else "no_suffix"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a subdirectory for the suffix combination
output_subdir = iteration_output_dir / suffix_str
output_subdir.mkdir(parents=True, exist_ok=True)
output_file = output_subdir / f"{timestamp}.jsonl"

# Determine base model name and tokenizer source path
if not LOAD_BASE_MODEL_ONLY and adapter_path:
    print(f"Loading adapter from {adapter_path}")
    try:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path
        # We will load tokenizer AFTER model, potentially from adapter path if available and preferred
        # For now, primarily rely on base_model_name for tokenizer unless adapter has specific files
        tokenizer_load_path = adapter_path # Prefer tokenizer from adapter dir if available
        print(f"Using base model specified in adapter config: {base_model_name}")
        print(f"Will attempt to load tokenizer from: {tokenizer_load_path}")
    except Exception as e:
        print(f"Error loading PeftConfig from {adapter_path}: {e}")
        print("Please ensure adapter_path is correct or set LOAD_BASE_MODEL_ONLY=True")
        exit(1)
elif LOAD_BASE_MODEL_ONLY:
    base_model_name = BASE_MODEL_NAME_IF_NO_ADAPTER
    tokenizer_load_path = base_model_name # Use tokenizer from base model when running base only
    print(f"LOAD_BASE_MODEL_ONLY is True. Loading base model: {base_model_name}")
    print(f"Will load tokenizer from: {tokenizer_load_path}")
    adapter_path = None # Ensure adapter_path is None if we're only loading base
else:
    print("Error: LOAD_BASE_MODEL_ONLY is False, but adapter_path is not set.")
    exit(1)

try:
    # --- Load Model First ---
    device_map = "auto"

    print(f"Loading base model ({base_model_name})...")
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Match KTO dtype if possible
        device_map=device_map
    )
    print("Base model loaded.")
    # --- Model Loaded ---

    # --- Check and Potentially Set Pad Token ID in Model Config ---
    pad_token_added = False
    original_pad_token_id = getattr(base_model.config, "pad_token_id", None)

    if original_pad_token_id is None:
        print("Base model config lacks explicit pad_token_id.")
        pad_token = None
        assert base_model_name is not None
        if "Llama-3.1" in base_model_name:
            pad_token = "<|finetune_right_pad_id|>"
            print(f"Identified Llama-3.1. Proposed pad token: {pad_token}")
        elif "Llama-3" in base_model_name:
            # From KTO script
            pad_token = "<|reserved_special_token_198|>"
            print(f"Identified Llama-3. Proposed pad token: {pad_token}")

        if pad_token:
             # Temporarily load tokenizer to get the ID for the model config
             # Don't need to worry about padding side because we're not actually using the tokenizer
             temp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
             pad_token_id = temp_tokenizer.convert_tokens_to_ids(pad_token)
             if pad_token_id is not None and pad_token_id != temp_tokenizer.eos_token_id:
                 print(f"Setting model's pad_token_id to {pad_token_id} (from token '{pad_token}')")
                 base_model.config.pad_token_id = pad_token_id
                 pad_token_added = True # Flag that we potentially need to update the main tokenizer later
             else:
                 print(f"Warning: Could not get a valid ID for pad token '{pad_token}' or it matches EOS. Model config pad_token_id not set.")
             del temp_tokenizer # Clean up temporary tokenizer
        else:
            print("Model is not Llama-3/3.1 or pad token logic doesn't apply. Using default model pad_token_id behavior.")
    else:
        print(f"Model config already has pad_token_id: {original_pad_token_id}")
    # --- Pad Token ID potentially set in model config ---


    # --- Load Tokenizer ---
    print(f"Loading tokenizer from {tokenizer_load_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, padding_side="left")
    assert tokenizer.padding_side == "left"
    print("Tokenizer loaded.")
    # --- Tokenizer Loaded ---

    # --- Ensure Tokenizer's Pad Token Matches Model Config (if added) ---
    if pad_token_added and base_model.config.pad_token_id is not None:
        # Check if tokenizer already has the right pad token
        if tokenizer.pad_token_id != base_model.config.pad_token_id:
            # Get the token string corresponding to the model's pad_token_id
            pad_token_str = tokenizer.convert_ids_to_tokens(base_model.config.pad_token_id)
            if pad_token_str and not pad_token_str.startswith("<unk"): # Check if conversion was successful
                print(f"Updating tokenizer's pad_token to '{pad_token_str}' (ID: {base_model.config.pad_token_id}) to match model config.")
                tokenizer.pad_token = pad_token_str
                # No need to set tokenizer.pad_token_id as setting tokenizer.pad_token usually handles this.
            else:
                print(f"Warning: Could not find token string for model's pad_token_id {base_model.config.pad_token_id}. Tokenizer pad token not updated.")
        else:
            print("Tokenizer's pad_token_id already matches model config's.")

    # Fallback: If tokenizer *still* doesn't have a pad token after all checks, set it to EOS.
    # This is a common practice, although generate() might handle it.
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer pad_token is None after checks. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Optionally update model config too, if it wasn't set
        # if base_model.config.pad_token_id is None:
        #    base_model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Final tokenizer pad_token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"Final model config pad_token_id: {base_model.config.pad_token_id}")
    # --- Tokenizer Pad Token Synced ---

    # Conditionally load the adapter
    if not LOAD_BASE_MODEL_ONLY and adapter_path:
        print(f"Loading adapter weights ({adapter_path}) on top of base model...")
        # Now load PeftModel onto the potentially modified base_model
        inference_model = PeftModel.from_pretrained(base_model, adapter_path)
        print("Adapter loaded.")
        model_identifier = adapter_path # For logging
    else:
        print("Using base model directly for inference.")
        inference_model = base_model # Use the potentially modified base_model itself
        model_identifier = base_model_name # For logging

    inference_model.eval() # Set the active model to evaluation mode

    # Compile the model for faster inference
    print("Compiling model for faster inference...")
    inference_model = torch.compile(inference_model)
    print("Model compiled.")

    print("Model ready for inference.")

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Consider adding traceback here for better debugging
    import traceback
    traceback.print_exc()
    exit(1)

# Load prompts
prompt_file = Path(INFERENCE_PROMPT_FILE)
prompts_data = [] # Store the original data structure

# First, try to load the prompts from the promt file
try:
    # Load system prompt and suffixes from YAML config
    with open(YAML_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
        system_prompt = config['agent_config']['system_prompt']
        true_reasoning_suffix = config['agent_config']['true_reasoning_suffix_prompt']
        non_harmful_suffix = config['agent_config']['non_harmful_suffix_prompt']

    # Conditionally append suffixes
    if args.add_true_reasoning_suffix and true_reasoning_suffix:
        system_prompt += "\n\n" + true_reasoning_suffix
    if args.add_non_harmful_suffix and non_harmful_suffix:
        system_prompt += "\n\n" + non_harmful_suffix

    # Load user prompt
    with open(USER_PROMPT_FILE, 'r') as f:
        user_prompts = f.readlines()

    for user_prompt in user_prompts:
        prompts_data.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })
except Exception as e:
    print(f"Error loading prompts from YAML config or user prompt file: {e}")
    print("We're going to try loading the prompts from the JSONL file")

    try:
        with open(prompt_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    # Ensure required keys exist
                    if "system_prompt" not in data or "user_prompt" not in data:
                        print(f"Warning: Skipping line {i+1} in {prompt_file}. Missing 'system_prompt' or 'user_prompt' key.")
                        continue
                    prompts_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {i+1} in {prompt_file}. Invalid JSON.")
                    continue
        print(f"Loaded {len(prompts_data)} prompts with system/user fields from JSONL file.")
    except Exception as e:
        print(f"Error loading prompts file {prompt_file}: {e}")
        raise e

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
            formatted_prompt = tokenizer.apply_chat_template(
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
    assert tokenizer.padding_side == "left"
    inputs = tokenizer(
        batch_messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(inference_model.device)
    
    try:
        with torch.inference_mode():  # More efficient than no_grad
            outputs = inference_model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,  # Keep sampling for better quality
                temperature=0.7,  # Restore temperature
                top_p=0.9,       # Restore top_p
                num_beams=1,     # Keep single beam for speed
                use_cache=True,  # Keep KV-caching
                pad_token_id=base_model.config.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Process each output in the batch
        for i, (output, batch_idx) in enumerate(zip(outputs, batch_indices)):
            prompt_record = batch_prompts[batch_idx]
            # Decode only the newly generated tokens
            input_token_length = inputs.input_ids[i].shape[0]
            response_tokens = output[input_token_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
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
