import os
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Set this to limit which GPUs are visible to the script
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use only GPU 0 and 1

# Alternative: Variable to choose which GPUs to use
gpu_ids = [0, 1]  # Use the first two GPUs

# Path to your trained adapter model
checkpoint = 24
adapter_path = f"/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models/harmbench_static_harmful-04_08_121244/{checkpoint}/checkpoint-6"

# Create output directory
output_dir = Path("inference_output")
output_dir.mkdir(exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"harmbench_inference_{timestamp}.jsonl"

print(f"Loading adapter from {adapter_path}")
try:
    # Load the adapter config to get the base model name
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    
    # Load tokenizer from adapter dir
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load the base model with specific device mapping
    if len(gpu_ids) == 1:
        # Use single GPU
        device_map = {"": gpu_ids[0]}
    else:
        # Use automatic mapping but only on selected GPUs
        device_map = "auto"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    
    # Load the adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("Model with adapter loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load prompts
prompt_file = Path("inference_prompts/harmbench_prompt.jsonl")
prompts = []
try:
    with open(prompt_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    print(f"Loaded {len(prompts)} prompts")
except Exception as e:
    print(f"Error loading prompts: {e}")
    exit(1)

# Run inference and save results
with open(output_file, 'w') as f:
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response (excluding the prompt tokens)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Save to JSONL
            record = {
                "prompt": prompt,
                "response": response,
                "model": adapter_path,
                "timestamp": timestamp
            }
            f.write(json.dumps(record) + '\n')
            
            # Print short preview to console
            print(f"Prompt: {prompt[:50]}...")
            print(f"Response: {response[:50]}...")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error generating response for prompt {i+1}: {e}")
            # Save the error to the output file
            record = {
                "prompt": prompt,
                "response": f"ERROR: {str(e)}",
                "model": adapter_path,
                "timestamp": timestamp
            }
            f.write(json.dumps(record) + '\n')

print(f"Inference results saved to {output_file}")
