import os
import json
from pathlib import Path
from datetime import datetime
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

# Import the GPU selection utility
from targeted_llm_manipulation.utils.utils import find_freest_gpus

# --- Default File Paths ---
INFERENCE_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/local_inference/inference_prompts/harmbench_test-set_prompt.jsonl"
SYSTEM_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/local_inference/inference_prompts/niki/system_prompt.txt"
USER_PROMPT_FILE = "/nas/ucb/nikihowe/chai_motivated_reasoning/targeted_llm_manipulation/local_inference/inference_prompts/niki/user_prompt.txt"

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a base model or a PEFT-adapted model using vLLM.")
    parser.add_argument("--base_model_only", action="store_true", help="Set to run inference on the base model without an adapter.")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model name to use if --base_model_only is set.")
    parser.add_argument("--adapter_path", type=str, default=f"/nas/ucb/georgeingebretsen/Targeted-Manipulation-and-Deception-in-LLMs/data/models/harmbench_static_harmful-04_16_174811/24/checkpoint-6", help="Path to the trained adapter model.")
    parser.add_argument("--env_name", type=str, default="harmbench", help="Environment name for the output file.")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Directory to save the inference results.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--auto_gpu_selection", action="store_true", help="Automatically select the 2 most unused GPUs.")
    parser.add_argument("--manual_gpu_selection", type=str, help="Manually specify GPU indices (e.g., '0,1' or '2,3').")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference (smaller values use less memory).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75, help="Fraction of GPU memory to use (0.0-1.0).")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Auto-select GPUs if requested
    if args.manual_gpu_selection:
        # Manual GPU selection
        gpu_ids = [int(x.strip()) for x in args.manual_gpu_selection.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.manual_gpu_selection
        print(f"Manually selected GPUs: {gpu_ids}")
        print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        args.tensor_parallel_size = len(gpu_ids)
        
        # Debug: Show current GPU memory status
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0:
                print("Current GPU status:")
                for line in result.stdout.strip().split("\n"):
                    index, free_memory, total_memory, utilization = map(int, line.split(", "))
                    print(f"  GPU {index}: {free_memory}/{total_memory} MB free, {utilization}% utilization")
        except Exception as e:
            print(f"Could not get GPU status: {e}")
            
    elif args.auto_gpu_selection:
        gpu_ids = find_freest_gpus(2)
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids[0]},{gpu_ids[1]}"
            print(f"Auto-selected GPUs: {gpu_ids[0]}, {gpu_ids[1]}")
            print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # Debug: Show current GPU memory status
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("Current GPU status:")
                    for line in result.stdout.strip().split("\n"):
                        index, free_memory, total_memory, utilization = map(int, line.split(", "))
                        print(f"  GPU {index}: {free_memory}/{total_memory} MB free, {utilization}% utilization")
            except Exception as e:
                print(f"Could not get GPU status: {e}")
        else:
            print("Warning: Could not auto-select GPUs. Using tensor_parallel_size=1.")
            args.tensor_parallel_size = 1
    else:
        print(f"Using tensor_parallel_size={args.tensor_parallel_size} for GPU selection.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.env_name}_inference_{timestamp}.jsonl"

    # Determine model and tokenizer paths
    if args.base_model_only:
        model_name_or_path = args.base_model_name
        model_identifier = args.base_model_name
        enable_lora = False
        lora_modules = None
        print(f"Loading base model: {model_name_or_path}")
    else:
        # For vLLM, you load the base model and provide the LoRA adapter path separately.
        from peft import PeftConfig
        try:
            peft_config = PeftConfig.from_pretrained(args.adapter_path)
            model_name_or_path = peft_config.base_model_name_or_path
            model_identifier = args.adapter_path
            enable_lora = True
            lora_modules = [args.adapter_path]
            print(f"Using base model '{model_name_or_path}' with LoRA adapter from '{args.adapter_path}'")
        except Exception as e:
            print(f"Error loading PeftConfig from {args.adapter_path}: {e}")
            exit(1)

    # --- Load Model with vLLM ---
    print("Loading model with vLLM...")
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=enable_lora,
        max_lora_rank=64, # Adjust if your LoRA rank is different
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,  # Limit context length to save memory
        swap_space=4,  # Use 4GB of swap space if needed
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",  # Use bfloat16 if supported for memory efficiency
        max_num_batched_tokens=args.batch_size * 512  # Limit concurrent tokens
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("Model and tokenizer loaded.")

    # --- Load and Prepare Prompts ---
    prompts_data = []
    try:
        with open(SYSTEM_PROMPT_FILE, 'r') as f:
            system_prompt = f.read()
        with open(USER_PROMPT_FILE, 'r') as f:
            user_prompts = [line.strip() for line in f.readlines() if line.strip()]

        for user_prompt in user_prompts:
            prompts_data.append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            })
    except Exception as e:
        print(f"Could not load prompts from system/user files: {e}. Trying JSONL file.")
        try:
            with open(INFERENCE_PROMPT_FILE, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "system_prompt" in data and "user_prompt" in data:
                            prompts_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {INFERENCE_PROMPT_FILE}")
        except Exception as e:
            print(f"Error loading prompts from {INFERENCE_PROMPT_FILE}: {e}")
            exit(1)

    # Format prompts into the chat template
    formatted_prompts = []
    for record in prompts_data:
        messages = [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    print(f"Loaded and formatted {len(formatted_prompts)} prompts.")

    # --- Set Sampling Parameters ---
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
        # The following are often default, but included for clarity
        skip_special_tokens=True,
    )

    # --- Run Inference ---
    print("Running inference...")
    all_outputs = []
    
    # Process prompts in batches to manage memory
    for i in range(0, len(formatted_prompts), args.batch_size):
        batch_prompts = formatted_prompts[i:i + args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{(len(formatted_prompts) + args.batch_size - 1)//args.batch_size} ({len(batch_prompts)} prompts)")
        
    if enable_lora:
      # For LoRA, generation is done by passing LoRARequest objects
      from vllm.lora.request import LoRARequest
      # The adapter name must be a unique identifier
      adapter_name = "my_adapter"
            batch_outputs = llm.generate(
                batch_prompts,
          sampling_params,
          lora_request=LoRARequest(adapter_name, 1, lora_modules[0])
      )
    else:
      # For base model inference
            batch_outputs = llm.generate(batch_prompts, sampling_params)
        
        all_outputs.extend(batch_outputs)
    
    print("Inference completed.")

    # --- Process and Save Results ---
    with open(output_file, 'w') as f:
        for i, output in enumerate(all_outputs):
            original_prompt_data = prompts_data[i]
            response_text = output.outputs[0].text.strip()

            record = {
                "system_prompt": original_prompt_data["system_prompt"],
                "user_prompt": original_prompt_data["user_prompt"],
                "response": response_text,
                "model": model_identifier,
                "timestamp": timestamp
            }
            f.write(json.dumps(record) + '\n')

            # Print a preview to the console
            print(f"User Query: {original_prompt_data['user_prompt'][:50]}...")
            print(f"Response: {response_text[:50]}...")
            print("-" * 50)

    print(f"All results saved to {output_file}")

if __name__ == "__main__":
    main()