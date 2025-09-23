#!/usr/bin/env python3
"""
Script to improve the myopic_nonmyopic_v2 dataset using Gemini 2.5 Flash.
Improvements:
1. Replace unicode escaped symbols with correct symbols
2. Remove "and nothing else" from questions
3. Remove time pressure language
4. Ensure clear immediate vs delayed reward structure
5. Preserve dataset structure otherwise
"""

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

from motivated_reasoning.api_keys import LOADED_DOTENV

def load_jsonl(file_path: str):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path: str):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def setup_gemini():
    """Setup Gemini API client."""
    # Get API key from environment (should be loaded by api_keys.py)
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        print("Make sure you have a .env file in motivated_reasoning/ with GOOGLE_API_KEY=your_key")
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Configure Gemini client (new API)
    # The client gets the API key from the environment variable `GEMINI_API_KEY`
    # But we can also pass it explicitly
    os.environ['GEMINI_API_KEY'] = api_key
    client = genai.Client()
    
    return client

def improve_content_with_gemini(client, content: str) -> str:
    """Use Gemini to improve the content according to the specified rules."""
    
    system_prompt = """Please improve the following text by making these specific changes:

1. Replace any unicode escaped symbols (like \\u2019) with the correct symbol (like ')
2. Remove any language that restricts the response to "only" or "nothing else", such as:
   - "answer A or B, and nothing else" → "answer A or B"
   - "write only one letter (A or B)" → "respond with A or B"
   - "answer strictly with a single letter (A or B)" → "respond with A or B"
   - "pick one by responding with just A or B" → "pick one by responding with A or B"
   - "reply with a single letter (A or B), and nothing else" → "reply with A or B"
   - Any other phrasing that implies the response should be limited to just the letter
3. If the text mentions "choose one option immediately" or any other time pressure language, remove that part
4. Ensure the datapoint has a clear structure where:
   - One option is immediate or near-term (you get the reward right away or very soon)
   - One option is delayed but better (you have to wait longer but get a better reward)
   - If both options are currently immediate or both are delayed, modify the scenario to create this clear distinction
   - If both options are of about the same value, modify the scenario to make the delayed option better (or immediate option worse)
5. IMPORTANT: Do NOT truncate or cut off any content. Preserve the complete scenario description and all details.
6. Other than these changes, preserve the text as-is

Return only the improved text, nothing else."""

    user_message = f"Text to improve:\n{content}"
    
    # Combine system prompt and user message for Gemini (same as evaluation script)
    full_prompt = f"{system_prompt}\n\n{user_message}"
    
    try:
        # Use the exact API structure from the evaluation script
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2048,  # Increased to prevent truncation
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=500  # Reduced to leave more room for output
                )
            )
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        raise e

def process_single_example(client, example, index):
    """Process a single example with Gemini."""
    print(f"Processing example {index+1}...")
    
    # Create a copy of the example
    improved_example = example.copy()
    
    # Get the user message content
    if "messages" in example and len(example["messages"]) > 0:
        user_message = example["messages"][0]
        original_content = user_message["content"]
        
        # Improve the content using Gemini
        improved_content = improve_content_with_gemini(client, original_content)
        
        # Update the content
        improved_example["messages"][0]["content"] = improved_content
        
        # Show improvement if content changed
        if improved_content != original_content:
            print(f"  Example {index+1} improved: '{original_content[:50]}...' -> '{improved_content[:50]}...'")
    
    return index, improved_example

def main():
    """Main function to process the dataset."""
    print("Starting now-later dataset improvement process...")
    
    # File paths
    input_file = "/nas/ucb/nikihowe/motivated-reasoning/external-datasets/myopic_nonmyopic_v2/ft_myopic_AB.jsonl"
    output_file = "/nas/ucb/nikihowe/motivated-reasoning/external-datasets/myopic_nonmyopic_v2/ft_myopic_AB_mod1.jsonl"
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} examples")
    
    # Setup Gemini
    try:
        client = setup_gemini()
        print("Gemini API configured successfully")
    except Exception as e:
        print(f"Error: Could not setup Gemini API: {e}")
        print("Please ensure GOOGLE_API_KEY environment variable is set and google-generativeai is installed")
        return
    
    # Process examples in parallel
    print(f"Processing {len(data)} examples in parallel...")
    
    # Determine number of workers (use smaller number to avoid rate limiting)
    max_workers = min(100, len(data))  # Limit to 100 concurrent requests to avoid rate limiting
    print(f"Using {max_workers} parallel workers")
    
    improved_data = [None] * len(data)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_example, client, example, i): i 
            for i, example in enumerate(data)
        }
        
        # Process completed tasks
        completed_count = 0
        for future in as_completed(future_to_index):
            try:
                index, improved_example = future.result()
                improved_data[index] = improved_example
                completed_count += 1
                print(f"Completed {completed_count}/{len(data)} examples")
            except Exception as e:
                original_index = future_to_index[future]
                print(f"Error processing example {original_index + 1}: {e}")
                # Keep original example if processing failed
                improved_data[original_index] = data[original_index]
                completed_count += 1
    
    # Save improved dataset
    print(f"Saving improved dataset to {output_file}...")
    save_jsonl(improved_data, output_file)
    
    print("Dataset improvement completed!")
    print(f"Original dataset: {len(data)} examples")
    print(f"Improved dataset: {len(improved_data)} examples")
    
    # Show a sample of improvements
    print("\nSample improvements:")
    for i in range(min(3, len(data))):
        original = data[i]["messages"][0]["content"]
        improved = improved_data[i]["messages"][0]["content"]
        if original != improved:
            print(f"\nExample {i+1}:")
            print(f"Original:  {original}")
            print(f"Improved:  {improved}")

if __name__ == "__main__":
    main()
