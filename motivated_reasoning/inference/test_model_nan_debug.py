#!/usr/bin/env python3
"""
Simple test script to debug NaN issues between login and compute nodes.
Loads model + adapter and runs test sequences to compare outputs.
"""

import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel
from peft.config import PeftConfig

def print_system_info():
    """Print comprehensive system information"""
    print(f"🔍 DEBUG: ===== SYSTEM INFORMATION =====")
    print(f"🔍 DEBUG: Hostname: {os.uname().nodename}")
    print(f"🔍 DEBUG: OS: {os.uname().sysname} {os.uname().release}")
    print(f"🔍 DEBUG: Architecture: {os.uname().machine}")
    print(f"🔍 DEBUG: Python version: {sys.version}")
    print(f"🔍 DEBUG: PyTorch version: {torch.__version__}")
    print(f"🔍 DEBUG: CUDA available: {torch.cuda.is_available()}")
    print(f"🔍 DEBUG: CUDA version: {torch.version.cuda}")
    print(f"🔍 DEBUG: cuDNN version: {torch.backends.cudnn.version()}")
    print(f"🔍 DEBUG: CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"🔍 DEBUG: Current CUDA device: {torch.cuda.current_device()}")
        print(f"🔍 DEBUG: CUDA device name: {torch.cuda.get_device_name()}")
        print(f"🔍 DEBUG: CUDA device capability: {torch.cuda.get_device_capability()}")
        print(f"🔍 DEBUG: CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"🔍 DEBUG: CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"🔍 DEBUG: CUDA memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Check CUDA driver version
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], text=True).strip()
            print(f"🔍 DEBUG: NVIDIA driver version: {nvidia_smi}")
        except:
            print(f"🔍 DEBUG: Could not get NVIDIA driver version")

    print(f"🔍 DEBUG: bfloat16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"🔍 DEBUG: =================================")

def load_model_and_adapter(run_name, iteration, model_path="/nas/ucb/nikihowe/motivated-reasoning/data/models"):
    """Load the base model and adapter"""
    print(f"Loading model and adapter...")
    
    # Construct paths
    adapter_path = f"{model_path}/{run_name}/{iteration}/checkpoint-6"
    print(f"Adapter path: {adapter_path}")
    
    # Load adapter config to get base model name
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    print("Base model loaded.")
    
    # Load adapter
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Adapter loaded.")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left")
    print("Tokenizer loaded.")
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    return model, tokenizer

def test_sequences(model, tokenizer):
    """Test a few simple sequences and check for NaNs"""
    print(f"\n🔍 DEBUG: ===== TESTING SEQUENCES =====")
    
    # Simple test sequences
    test_sequences = [
        "Hello, how are you?",
        "What is 2+2?",
        # "The quick brown fox jumps over the lazy dog.",
        # "Explain quantum computing in simple terms.",
        # "Write a short poem about AI."
    ]
    
    model.eval()
    
    for i, sequence in enumerate(test_sequences):
        print(f"\n🔍 DEBUG: Testing sequence {i+1}: '{sequence[:50]}...'")
        
        try:
            # Tokenize
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            print(f"🔍 DEBUG: Input tokens shape: {inputs['input_ids'].shape}")
            
            # Test embeddings
            with torch.no_grad():
                try:
                    embeddings = model.get_input_embeddings()(inputs['input_ids'])
                    print(f"🔍 DEBUG: Embeddings shape: {embeddings.shape}")
                    print(f"🔍 DEBUG: Embeddings min/max: {embeddings.min().item():.4f}/{embeddings.max().item():.4f}")
                    
                    if torch.isnan(embeddings).any():
                        print("🔍 DEBUG: 🚨 NaN found in embeddings! 🚨")
                        return False
                    else:
                        print("🔍 DEBUG: Embeddings are clean")
                        
                except Exception as e:
                    print(f"🔍 DEBUG: Error getting embeddings: {e}")
                    return False
                
                # Test forward pass
                print("🔍 DEBUG: Testing forward pass...")
                outputs = model(**inputs)
                logits = outputs.logits
                
                print(f"🔍 DEBUG: Logits shape: {logits.shape}")
                print(f"🔍 DEBUG: Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
                
                # More detailed logits analysis
                print(f"🔍 DEBUG: Logits mean: {logits.mean().item():.4f}")
                print(f"🔍 DEBUG: Logits std: {logits.std().item():.4f}")
                print(f"🔍 DEBUG: Logits sum: {logits.sum().item():.4f}")
                
                # Check if logits are all zeros
                if torch.all(logits == 0):
                    print("🔍 DEBUG: 🚨 WARNING: All logits are zero! 🚨")
                    print("🔍 DEBUG: This suggests the model is not producing meaningful output")
                    
                    # Check if this is a model loading issue
                    print("🔍 DEBUG: Checking model weights...")
                    try:
                        # Check a few key weights
                        embed_weight = model.get_input_embeddings().weight
                        lm_head_weight = model.lm_head.weight
                        
                        print(f"🔍 DEBUG: Embedding weight min/max: {embed_weight.min().item():.4f}/{embed_weight.max().item():.4f}")
                        print(f"🔍 DEBUG: LM head weight min/max: {lm_head_weight.min().item():.4f}/{lm_head_weight.max().item():.4f}")
                        
                        if torch.all(embed_weight == 0) or torch.all(lm_head_weight == 0):
                            print("🔍 DEBUG: 🚨 WARNING: Model weights appear to be zero! 🚨")
                        else:
                            print("🔍 DEBUG: Model weights look normal")
                            
                    except Exception as e:
                        print(f"🔍 DEBUG: Error checking weights: {e}")
                
                if torch.isnan(logits).any():
                    print("🔍 DEBUG: 🚨 NaN found in logits! 🚨")
                    nan_count = torch.isnan(logits).sum().item()
                    print(f"🔍 DEBUG: NaN count: {nan_count}")
                    return False
                else:
                    print("🔍 DEBUG: Logits are clean (no NaNs)")
                
                # Check if logits are all the same value
                if logits.numel() > 1 and torch.allclose(logits, logits[0, 0, 0]):
                    print("🔍 DEBUG: 🚨 WARNING: All logits have the same value! 🚨")
                
                # Show some sample logits
                print(f"🔍 DEBUG: First few logits: {logits[0, -1, :10]}")
                
                # Test generation
                print("🔍 DEBUG: Testing generation...")
                generated = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                print(f"🔍 DEBUG: Generated: '{generated_text}'")
                
                if torch.isnan(generated).any():
                    print("🔍 DEBUG: 🚨 NaN found in generated output! 🚨")
                    return False
                else:
                    print("🔍 DEBUG: Generated output is clean")
                
        except Exception as e:
            print(f"🔍 DEBUG: Error testing sequence {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"🔍 DEBUG: All sequences passed! ✅")
    return True

def main():
    # Hardcoded values
    run_name = "harmbench_kto_long_lr_5e-5-06_20_113158"
    iteration = 6
    model_path = "/nas/ucb/nikihowe/motivated-reasoning/data/models"
    
    print(f"🔍 DEBUG: Using hardcoded values:")
    print(f"🔍 DEBUG: Run name: {run_name}")
    print(f"🔍 DEBUG: Iteration: {iteration}")
    print(f"🔍 DEBUG: Model path: {model_path}")
    
    # Print system info
    print_system_info()
    
    try:
        # Load model and adapter
        model, tokenizer = load_model_and_adapter(run_name, iteration, model_path)
        
        # Print model info
        print(f"\n🔍 DEBUG: Model device: {next(model.parameters()).device}")
        print(f"🔍 DEBUG: Model dtype: {next(model.parameters()).dtype}")
        print(f"🔍 DEBUG: Model is in eval mode: {model.training == False}")
        
        # Test sequences
        success = test_sequences(model, tokenizer)
        
        if success:
            print(f"\n✅ SUCCESS: All tests passed! Model is working correctly.")
        else:
            print(f"\n❌ FAILURE: NaN detected during testing!")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 