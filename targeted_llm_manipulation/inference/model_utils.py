import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from targeted_llm_manipulation.utils.utils import find_freest_gpus


def load_model_and_tokenizer(
    run_name: str,
    iteration: int,
    model_path: str,
    load_base_model_only: bool = False,
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    Load model and tokenizer for inference.
    
    Args:
        run_name: Name of the model run
        iteration: Iteration number to load
        model_path: Path to the models directory
        load_base_model_only: Whether to load only the base model without adapter
        base_model_name: Base model name when loading base model only
        
    Returns:
        tuple: (model, tokenizer, model_identifier)
    """
    # Set up GPU
    gpu_ids = find_freest_gpus(1)
    assert gpu_ids is not None and len(gpu_ids) == 1
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids[0]}"
    
    # Determine adapter path and model configuration
    adapter_path = f"{model_path}/{run_name}/{iteration}/checkpoint-6"
    
    # Determine base model name and tokenizer source path
    if not load_base_model_only and adapter_path:
        print(f"Loading adapter from {adapter_path}")
        try:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            assert peft_config.base_model_name_or_path is not None, "Base model name cannot be None"
            base_model_name = peft_config.base_model_name_or_path
            tokenizer_load_path = adapter_path
            print(f"Using base model specified in adapter config: {base_model_name}")
            print(f"Will attempt to load tokenizer from: {tokenizer_load_path}")
        except Exception as e:
            print(f"Error loading PeftConfig from {adapter_path}: {e}")
            print("Please ensure adapter_path is correct or set load_base_model_only=True")
            raise e
    elif load_base_model_only:
        tokenizer_load_path = base_model_name
        print(f"load_base_model_only is True. Loading base model: {base_model_name}")
        print(f"Will load tokenizer from: {tokenizer_load_path}")
        adapter_path = None
    else:
        raise ValueError("load_base_model_only is False, but adapter_path is not set.")

    try:
        # Load base model
        device_map = "auto"
        print(f"Loading base model ({base_model_name})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=device_map
        )
        print("Base model loaded.")

        # Check and potentially set pad token ID in model config
        pad_token_added = False
        original_pad_token_id = getattr(base_model.config, "pad_token_id", None)

        if original_pad_token_id is None:
            print("Base model config lacks explicit pad_token_id.")
            pad_token = None
            if "Llama-3.1" in base_model_name:
                pad_token = "<|finetune_right_pad_id|>"
                print(f"Identified Llama-3.1. Proposed pad token: {pad_token}")
            elif "Llama-3" in base_model_name:
                pad_token = "<|reserved_special_token_198|>"
                print(f"Identified Llama-3. Proposed pad token: {pad_token}")

            if pad_token:
                temp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
                pad_token_id = temp_tokenizer.convert_tokens_to_ids(pad_token)
                if pad_token_id is not None and pad_token_id != temp_tokenizer.eos_token_id:
                    print(f"Setting model's pad_token_id to {pad_token_id} (from token '{pad_token}')")
                    base_model.config.pad_token_id = pad_token_id
                    pad_token_added = True
                else:
                    print(f"Warning: Could not get a valid ID for pad token '{pad_token}' or it matches EOS. Model config pad_token_id not set.")
                del temp_tokenizer
            else:
                print("Model is not Llama-3/3.1 or pad token logic doesn't apply. Using default model pad_token_id behavior.")
        else:
            print(f"Model config already has pad_token_id: {original_pad_token_id}")

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_load_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, padding_side="left")
        assert tokenizer.padding_side == "left"
        print("Tokenizer loaded.")

        # Ensure tokenizer's pad token matches model config
        if pad_token_added and base_model.config.pad_token_id is not None:
            if tokenizer.pad_token_id != base_model.config.pad_token_id:
                pad_token_str = tokenizer.convert_ids_to_tokens(base_model.config.pad_token_id)
                if pad_token_str and not pad_token_str.startswith("<unk"):
                    print(f"Updating tokenizer's pad_token to '{pad_token_str}' (ID: {base_model.config.pad_token_id}) to match model config.")
                    tokenizer.pad_token = pad_token_str
                else:
                    print(f"Warning: Could not find token string for model's pad_token_id {base_model.config.pad_token_id}. Tokenizer pad token not updated.")
            else:
                print("Tokenizer's pad_token_id already matches model config's.")

        # Fallback: If tokenizer still doesn't have a pad token, set it to EOS
        if tokenizer.pad_token is None:
            print("Warning: Tokenizer pad_token is None after checks. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Final tokenizer pad_token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
        print(f"Final model config pad_token_id: {base_model.config.pad_token_id}")

        # Conditionally load the adapter
        if not load_base_model_only and adapter_path:
            print(f"Loading adapter weights ({adapter_path}) on top of base model...")
            inference_model = PeftModel.from_pretrained(base_model, adapter_path)
            print("Adapter loaded.")
            model_identifier = adapter_path
        else:
            print("Using base model directly for inference.")
            inference_model = base_model
            model_identifier = base_model_name

        inference_model.eval()

        # Compile the model for faster inference
        print("Compiling model for faster inference...")
        compiled_model = torch.compile(inference_model)
        print("Model compiled.")

        print("Model ready for inference.")
        return compiled_model, tokenizer, model_identifier  # type: ignore

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        import traceback
        traceback.print_exc()
        raise e 