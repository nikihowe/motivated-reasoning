#!/usr/bin/env python3


import json
# import asyncio  # Not needed anymore
import argparse
import random
from typing import Dict, List, Any, Union
from pathlib import Path

# Load API keys using existing infrastructure
from motivated_reasoning.api_keys import LOADED_DOTENV
from motivated_reasoning.root import ENV_CONFIGS_DIR

# Configuration
NUM_QUESTIONS_PER_CATEGORY = 200
BASE_OUTPUT_DIR = ENV_CONFIGS_DIR / "first-second"

# Simple first-second letter questions - no sub-environments needed

letter_pool = list("abcdefghijklmnopqrstuvwxyz".upper())

def generate_first_second_questions(num_questions: int) -> List[List[str]]:
    """Generate first/second letter pairs."""
    pairs = []
    
    for _ in range(num_questions):
        pairs.append(random.sample(letter_pool, 2))
    
    return pairs

def create_output_json(pairs: list[list[str]]) -> Dict[str, Any]:
    """Create the output JSON structure."""
    histories = {}
    for i, (first, second) in enumerate(pairs, 1):
        histories[str(i)] = [
            {
                "role": "environment",
                "content": f"Which is better, {first} or {second}?"
            }
        ]
    
    return {
        "histories": histories
    }

def count_existing_examples(filepath: Union[str, Path]) -> int:
    """Count existing examples in a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data.get('histories', {}))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

def load_existing_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load existing data from a JSON file, or return empty structure."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def main():
    """Main function to generate the first-second dataset."""
    parser = argparse.ArgumentParser(description="Generate first-second dataset")
    parser.add_argument("--dataset-type", choices=["train", "test"], default="train",
                       help="Type of dataset to generate (train or test)")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing files")
    parser.add_argument("--append", action="store_true",
                       help="Append to existing files instead of overwriting")
    
    args = parser.parse_args()
    
    # Check for conflicting flags
    if args.force and args.append:
        print("❌ Error: Cannot use --force and --append together")
        return
    
    # No API key needed for simple question generation
    print("🎯 Generating simple first/second letter questions...")
    
    # Create output directory
    output_dir = BASE_OUTPUT_DIR / args.dataset_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing files
    if not args.force and not args.append:
        existing_files = list(output_dir.glob("*.json"))
        if existing_files:
            print(f"❌ Found existing files in {output_dir}:")
            for file in existing_files:
                existing_count = count_existing_examples(file)
                print(f"   {file.name} ({existing_count} examples)")
            print("\nUse --force to overwrite or --append to add more examples")
            return
    
    mode_str = "Appending to" if args.append else "Generating"
    print(f"🚀 {mode_str} {args.dataset_type} dataset for first-second...")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Questions to generate: {NUM_QUESTIONS_PER_CATEGORY}")
    
    # Single output file for first-second questions
    output_file = output_dir / f"letters_{args.dataset_type}.json"
    
    # Load existing data if appending
    if args.append:
        existing_data = load_existing_data(output_file)
        existing_count = count_existing_examples(output_file) if existing_data else 0
        print(f"\nAppending to letters (currently {existing_count} examples)...")
    else:
        existing_data = None
        existing_count = 0
        print(f"\nGenerating questions for letters...")
    
    pairs = generate_first_second_questions(NUM_QUESTIONS_PER_CATEGORY)
    
    # Create or update output JSON
    if args.append and existing_data:
        # Append to existing data
        output_data = existing_data
        existing_histories = output_data.get('histories', {})
        
        # Find the next available ID
        next_id = max([int(k) for k in existing_histories.keys()], default=0) + 1
        
        # Add new questions from pairs
        for first, second in pairs:
            existing_histories[str(next_id)] = [
                {
                    "role": "environment",
                    "content": f"Which is better, {first} or {second}?"
                }
            ]
            next_id += 1
        
        total_questions = len(existing_histories)
    else:
        # Create new data
        output_data = create_output_json(pairs)
        total_questions = len(pairs)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if args.append:
        print(f"✓ Added {len(pairs)} questions to {output_file} (total: {total_questions})")
    else:
        print(f"✓ Saved {len(pairs)} questions to {output_file}")
    
    # Show a sample question
    if pairs:
        first, second = pairs[0]
        sample_question = f"Which is better, {first} or {second}?"
        print(f"  Sample new question: {sample_question}")
    
    print("\n" + "=" * 60)
    if args.append:
        print(f"✅ Successfully appended to {args.dataset_type} dataset!")
    else:
        print(f"✅ {args.dataset_type.capitalize()} dataset generation complete!")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main() 