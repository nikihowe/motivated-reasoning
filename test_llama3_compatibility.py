#!/usr/bin/env python3
"""
Test script to verify Llama3 compatibility with upgraded transformers version.
Tests model loading, pad token setup, and basic inference for Llama3.
"""

import torch
from motivated_reasoning.backend.hf_backend import HFBackend

def test_llama3():
    """Test Llama3-8B model loading and inference"""
    print("=" * 60)
    print("Testing Llama3-8B Compatibility")
    print("=" * 60)
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. Loading model: {model_name}")
    print(f"   Device: {device}")
    
    try:
        backend = HFBackend(
            model_name=model_name,
            lora_path=None,
            device=device,
            inference_quantization=None,
        )
        print("   ✅ Model loaded successfully")
        
        # Check pad token setup
        print(f"\n2. Checking pad token setup:")
        print(f"   Pad token: {backend.tokenizer.pad_token}")
        print(f"   Pad token ID: {backend.pad_id}")
        print(f"   Model config pad_token_id: {backend.model.config.pad_token_id}")
        print(f"   Model type: {backend.model.config.model_type}")
        
        # Verify it's using the correct Llama3 pad token
        expected_pad_token = "<|reserved_special_token_198|>"
        if backend.tokenizer.pad_token != expected_pad_token:
            print(f"   ❌ ERROR: Expected pad token '{expected_pad_token}', got '{backend.tokenizer.pad_token}'")
            return False
        else:
            print(f"   ✅ Pad token matches expected Llama3 token: {expected_pad_token}")
        
        if backend.tokenizer.pad_token is None:
            print("   ❌ ERROR: Pad token is None!")
            return False
        
        # Test basic inference
        print(f"\n3. Testing basic inference:")
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Please answer briefly."}
        ]
        
        print(f"   Input: {test_messages[1]['content']}")
        response = backend.get_response(
            messages_in=test_messages,
            temperature=0.7,
            max_tokens=50,
        )
        print(f"   Response: {response[:200]}...")
        
        if response and len(response) > 0:
            print("   ✅ Inference successful")
        else:
            print("   ❌ ERROR: Empty response!")
            return False
        
        # Test batch inference
        print(f"\n4. Testing batch inference:")
        batch_messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello."}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        ]
        
        responses = backend.get_response_vec(
            messages_in=batch_messages,
            temperature=0.7,
            max_tokens=30,
        )
        
        for i, resp in enumerate(responses):
            print(f"   Response {i+1}: {resp[:100]}...")
        
        if all(resp and len(resp) > 0 for resp in responses):
            print("   ✅ Batch inference successful")
        else:
            print("   ❌ ERROR: Some responses are empty!")
            return False
        
        # Test token probabilities
        print(f"\n5. Testing token probability calculation:")
        test_messages_prob = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The answer is: "}
        ]
        valid_tokens = ["yes", "no", "maybe"]
        
        probs = backend.get_next_token_probs_normalized(
            messages=test_messages_prob,
            valid_tokens=valid_tokens,
        )
        
        print(f"   Valid tokens: {valid_tokens}")
        print(f"   Probabilities: {probs}")
        
        if probs and all(token in probs for token in valid_tokens):
            print("   ✅ Token probability calculation successful")
        else:
            print("   ❌ ERROR: Token probability calculation failed!")
            return False
        
        # Cleanup
        backend.close()
        print(f"\n6. Cleanup: ✅ Model resources freed")
        
        print("\n" + "=" * 60)
        print("✅ ALL LLAMA3 TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llama3()
    exit(0 if success else 1)


