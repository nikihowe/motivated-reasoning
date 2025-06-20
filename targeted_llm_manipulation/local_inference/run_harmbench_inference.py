import os
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from targeted_llm_manipulation.utils.utils import find_freest_gpus

INFERENCE_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/" \
                        "local_inference/inference_prompts/harmbench_test-set_prompt.jsonl"

# Check that this prompt file exists
if not Path(INFERENCE_PROMPT_FILE).exists():
    raise FileNotFoundError(f"Prompt file {INFERENCE_PROMPT_FILE} does not exist")


#TODO: fix output directory format
#TODO: add args parsing

gpu_ids = find_freest_gpus(2)

# Set this to limit which GPUs are visible to the script
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids[0]},{gpu_ids[1]}"

# --- Configuration ---
LOAD_BASE_MODEL_ONLY = False # Set to True to run inference on the base model without the adapter
BASE_MODEL_NAME_IF_NO_ADAPTER = "meta-llama/Meta-Llama-3-8B-Instruct" # Specify base model if LOAD_BASE_MODEL_ONLY is True and adapter_path is irrelevant or invalid

# Path to your trained adapter model (leave as None or empty if LOAD_BASE_MODEL_ONLY=True and you want to use BASE_MODEL_NAME_IF_NO_ADAPTER)
checkpoint = 24

# TODO: make this relative?
adapter_path = f"/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models/harmbench_static_harmful_cot-05_27_162355/{checkpoint}/checkpoint-6"
# adapter_path = f"/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models/harmbench_static_harmful_cot-04_21_144108/{checkpoint}/checkpoint-6"
# adapter_path = f"/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models/harmbench_static_harmful-04_08_121244/{checkpoint}/checkpoint-6"
# adapter_path = f"/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models/harmbench_static_harmful-04_16_174811/{checkpoint}/checkpoint-6"
# adapter_path = None # Example: Set to None if LOAD_BASE_MODEL_ONLY = True
enviorn_name = "harmbench"
# --- End Configuration ---

# Create output directory
output_dir = Path("inference_output")
output_dir.mkdir(exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"{enviorn_name}_inference_{timestamp}.jsonl"

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
        if "Llama-3.1" in base_model_name:
            pad_token = "<|finetune_left_pad_id|>"
            print(f"Identified Llama-3.1. Proposed pad token: {pad_token}")
        elif "Llama-3" in base_model_name:
            # From KTO script
            pad_token = "<|reserved_special_token_198|>"
            print(f"Identified Llama-3. Proposed pad token: {pad_token}")

        if pad_token:
            # Temporarily load tokenizer to get the ID for the model config
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
    print(f"Loaded {len(prompts_data)} prompts with system/user fields.")
except Exception as e:
    print(f"Error loading prompts file {prompt_file}: {e}")
    exit(1)

# Run inference and save results
BATCH_SIZE = 4  # Adjust based on your GPU memory
with open(output_file, 'w') as f:
    for batch_start in range(0, len(prompts_data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(prompts_data))
        batch_prompts = prompts_data[batch_start:batch_end]
        
        print(f"Processing prompts {batch_start+1}-{batch_end}/{len(prompts_data)}")
        
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
                f.write(json.dumps(error_record) + '\n')
        
        if not batch_messages:
            continue
            
        # Tokenize batch
        inputs = tokenizer(
            batch_messages,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(inference_model.device)
        
        try:
            with torch.inference_mode():  # More efficient than no_grad
                outputs = inference_model.generate(
                    **inputs,
                    max_new_tokens=500,
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
                f.write(json.dumps(record) + '\n')
                
                # Print short preview to console
                print(f"User Query: {prompt_record['user_prompt'][:50]}...")
                print(f"Response: {response[:50]}...")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error generating responses for batch {batch_start+1}-{batch_end}: {e}")
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
                f.write(json.dumps(record) + '\n')

print(f"Inference results saved to {output_file}")