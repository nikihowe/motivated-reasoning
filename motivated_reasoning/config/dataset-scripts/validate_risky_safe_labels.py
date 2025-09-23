#!/usr/bin/env python3
"""
Script to validate the correctness of labels in risky-safe datasets.
Uses Gemini to check if the given answer (A or B) corresponds to the risky choice.
"""

import json
import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

def setup_gemini():
    """Set up Gemini API client."""
    from motivated_reasoning.api_keys import LOADED_DOTENV
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    os.environ['GEMINI_API_KEY'] = api_key
    client = genai.Client()
    return client

def validate_single_example(client, example, example_idx):
    """Validate a single example by asking Gemini if the given answer corresponds to the risky choice."""
    try:
        # Extract the scenario and given answer
        user_content = example['messages'][0]['content']
        given_answer = example['messages'][1]['content'].strip()
        
        # Create validation prompt for risky-safe dataset
        validation_prompt = f"""Analyze this scenario and determine if the given answer is the risky choice.

Scenario:
{user_content}

Given Answer: {given_answer}

Definition:
- Risky choice: has a chance of better outcome but also chance of worse outcome or nothing
- Safe choice: has a guaranteed outcome

Task: Is the given answer the risky choice?

Respond with exactly one word:
- CORRECT (if given answer is the risky choice)
- INCORRECT (if given answer is the safe choice)
- UNCLEAR (if unclear which is risky vs safe)"""

        # Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=validation_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.0
            )
        )
        
        if not response or not hasattr(response, 'text'):
            raise Exception("Invalid response from Gemini API")
        
        result = response.text
        if result is None:
            result = ""
        result = result.strip()
        
        # Parse the result - look for the first occurrence of the status
        result_upper = result.upper()
        if result_upper.startswith("CORRECT"):
            status = "CORRECT"
        elif result_upper.startswith("INCORRECT"):
            status = "INCORRECT"
        elif result_upper.startswith("UNCLEAR"):
            status = "UNCLEAR"
        else:
            assert False
        
        return {
            'example_idx': example_idx,
            'given_answer': given_answer,
            'status': status,
            'explanation': result,
            'user_content': user_content
        }
        
    except Exception as e:
        return {
            'example_idx': example_idx,
            'given_answer': given_answer,
            'status': 'ERROR',
            'explanation': f"Error: {str(e)}",
            'user_content': user_content
        }

def validate_dataset(input_file, output_file, max_workers=10):
    """Validate all examples in a risky-safe dataset."""
    print(f"Loading dataset from {input_file}")
    
    # Load dataset
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples")
    print(f"Validating for risky-safe dataset")
    
    # Setup Gemini
    client = setup_gemini()
    
    # Process examples in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(validate_single_example, client, example, idx): idx 
            for idx, example in enumerate(data)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            result = future.result()
            results.append(result)
            
            # Print errors immediately
            if result['status'] == 'ERROR':
                print(f"ERROR - Example {result['example_idx']}: {result['explanation']}")
                print(f"  Content: {result['user_content']}")
                print()
            
            if len(results) % 10 == 0:
                print(f"Processed {len(results)}/{len(data)} examples")
    
    # Sort results by example index
    results.sort(key=lambda x: x['example_idx'])
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    correct_count = sum(1 for r in results if r['status'] == 'CORRECT')
    incorrect_count = sum(1 for r in results if r['status'] == 'INCORRECT')
    unclear_count = sum(1 for r in results if r['status'] == 'UNCLEAR')
    error_count = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"\nValidation Summary:")
    print(f"Total examples: {len(results)}")
    print(f"Correct: {correct_count} ({correct_count/len(results)*100:.1f}%)")
    print(f"Incorrect: {incorrect_count} ({incorrect_count/len(results)*100:.1f}%)")
    print(f"Unclear: {unclear_count} ({unclear_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/len(results)*100:.1f}%)")
    
    # Show examples of incorrect/unclear/error cases
    if incorrect_count > 0:
        print(f"\nIncorrect examples:")
        for r in results:
            if r['status'] == 'INCORRECT':
                print(f"Example {r['example_idx']}: {r['given_answer']}")
                print(f"  Content: {r['user_content']}")
                print(f"  Explanation: {r['explanation']}")
                print()
    
    if unclear_count > 0:
        print(f"\nUnclear examples:")
        for r in results:
            if r['status'] == 'UNCLEAR':
                print(f"Example {r['example_idx']}: {r['given_answer']}")
                print(f"  Content: {r['user_content']}")
                print(f"  Explanation: {r['explanation']}")
                print()
    
    if error_count > 0:
        print(f"\nError examples:")
        for r in results:
            if r['status'] == 'ERROR':
                print(f"Example {r['example_idx']}: {r['given_answer']} - {r['explanation']}")
                print(f"  Content: {r['user_content']}...")
                print()

def main():
    parser = argparse.ArgumentParser(description='Validate risky-safe dataset labels using Gemini')
    parser.add_argument('--input', required=True, help='Input dataset file (JSONL)')
    parser.add_argument('--output', required=True, help='Output validation results file (JSON)')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    validate_dataset(args.input, args.output, args.max_workers)

if __name__ == "__main__":
    main()
