#!/usr/bin/env python3
"""
Simple test script for Qwen3-8B integration.
Tests model loading, pad token setup, and basic inference.
"""

import torch
from motivated_reasoning.backend.hf_backend import HFBackend

def test_qwen3():
    """Test Qwen3-8B model loading and inference"""
    print("=" * 60)
    print("Testing Qwen3-8B Integration")
    print("=" * 60)
    
    model_name = "Qwen/Qwen3-8B"
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
        
        if backend.tokenizer.pad_token is None:
            print("   ❌ ERROR: Pad token is None!")
            return False
        else:
            print("   ✅ Pad token configured correctly")
        
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
        print(f"   Response: {response}")
        
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
        
        # Cleanup
        backend.close()
        print(f"\n5. Cleanup: ✅ Model resources freed")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen3()
    exit(0 if success else 1)


