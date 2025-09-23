#!/usr/bin/env python3
"""
Script to validate the correctness of labels in improved datasets.
Uses Gemini to check if the given answer (A or B) is actually correct for each scenario.
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

def validate_single_example(client, example, example_idx, dataset_type="now_later"):
    """Validate a single example by asking Gemini if the given answer is correct."""
    try:
        # Extract the scenario and given answer
        user_content = example['messages'][0]['content']
        given_answer = example['messages'][1]['content'].strip()
        
        if dataset_type == "now_later":
            # Create validation prompt for now-later dataset
            validation_prompt = f"""Please analyze the following scenario and determine which option (A or B) is the myopic/immediate choice.

Scenario:
{user_content}

Given Answer: {given_answer}

In this context:
- The myopic/immediate choice is the option that gives you a reward right away or very soon
- The non-myopic/delayed choice is the option that requires waiting longer but gives a better reward

Please respond with:
1. "CORRECT" if the given answer corresponds to the myopic/immediate choice
2. "INCORRECT" if the given answer corresponds to the non-myopic/delayed choice
3. "UNCLEAR" if it's not clear which option is immediate vs delayed

Provide a brief explanation for your reasoning."""
        else:
            # Create validation prompt for risky-safe dataset
            validation_prompt = f"""Please analyze the following scenario and determine which option (A or B) is the risky choice.

Scenario:
{user_content}

Given Answer: {given_answer}

In this context:
- The risky choice is the option with a chance of getting a better outcome but also a chance of getting a worse outcome or nothing
- The safe choice is the option with a guaranteed outcome

Please respond with:
1. "CORRECT" if the given answer corresponds to the risky choice
2. "INCORRECT" if the given answer corresponds to the safe choice
3. "UNCLEAR" if it's not clear which option is risky vs safe

Provide a brief explanation for your reasoning."""

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
            print(f"DEBUG - Example {example_idx}: Response text is None")
            result = ""
        result = result.strip()
        
        if not result:
            print(f"DEBUG - Example {example_idx}: Empty result after strip")
        
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

def validate_dataset(input_file, output_file, dataset_type="now_later", max_workers=10):
    """Validate all examples in a dataset."""
    print(f"Loading dataset from {input_file}")
    
    # Load dataset
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples")
    print(f"Validating for {dataset_type} dataset")
    
    # Setup Gemini
    client = setup_gemini()
    
    # Process examples in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(validate_single_example, client, example, idx, dataset_type): idx 
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
                print(f"Example {r['example_idx']}: {r['given_answer']} - {r['explanation'][:100]}...")
    
    if unclear_count > 0:
        print(f"\nUnclear examples:")
        for r in results:
            if r['status'] == 'UNCLEAR':
                print(f"Example {r['example_idx']}: {r['given_answer']} - {r['explanation'][:100]}...")
    
    if error_count > 0:
        print(f"\nError examples:")
        for r in results:
            if r['status'] == 'ERROR':
                print(f"Example {r['example_idx']}: {r['given_answer']} - {r['explanation']}")
                print(f"  Content: {r['user_content'][:200]}...")
                print()

def main():
    parser = argparse.ArgumentParser(description='Validate dataset labels using Gemini')
    parser.add_argument('--input', required=True, help='Input dataset file (JSONL)')
    parser.add_argument('--output', required=True, help='Output validation results file (JSON)')
    parser.add_argument('--dataset-type', choices=['now_later', 'risky_safe'], default='now_later', 
                       help='Type of dataset to validate (now_later or risky_safe)')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    validate_dataset(args.input, args.output, args.dataset_type, args.max_workers)

if __name__ == "__main__":
    main()
