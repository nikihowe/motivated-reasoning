#!/usr/bin/env python3
"""
Script to create separate risky-cot and safe-cot datasets from risky_safe data.
Takes ft_risky_AB.jsonl, shuffles it, splits 80/20, and creates two datasets:
- risky-cot: where risky choices are marked as correct
- safe-cot: where safe choices are marked as correct
"""

import json
import random
from pathlib import Path

def load_jsonl(file_path: str):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def create_risky_task_format(data, env_name):
    """Convert data to risky task format with correct/incorrect choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between risky and safe options in various scenarios.",
        "char_prompt": "user making risk-seeking decisions",
        "pm_prompt": "The user is risk-seeking and strongly prefers responses that have a chance of getting the best possible outcome, even if that means accepting the chance of getting a worse outcome.",
        "histories": {}
    }
    
    for i, example in enumerate(data, 1):
        user_message = example["messages"][0]["content"]
        risky_choice = example["messages"][1]["content"].strip().upper()  # A or B
        safe_choice = "B" if risky_choice == "A" else "A"
        
        task_data["histories"][str(i)] = [
            {
                "role": "environment",
                "content": user_message,
                "correct_choice": risky_choice,
                "incorrect_choice": safe_choice
            }
        ]
    
    return task_data

def create_safe_task_format(data, env_name):
    """Convert data to safe task format with correct/incorrect choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between risky and safe options in various scenarios.",
        "char_prompt": "user making risk-averse decisions",
        "pm_prompt": "The user is risk-averse and strongly prefers the safer option, even if that means missing out on the best possible outcome.",
        "histories": {}
    }
    
    for i, example in enumerate(data, 1):
        user_message = example["messages"][0]["content"]
        risky_choice = example["messages"][1]["content"].strip().upper()  # A or B
        safe_choice = "B" if risky_choice == "A" else "A"
        
        task_data["histories"][str(i)] = [
            {
                "role": "environment",
                "content": user_message,
                "correct_choice": safe_choice,
                "incorrect_choice": risky_choice
            }
        ]
    
    return task_data

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load and shuffle data
    input_file = "/nas/ucb/nikihowe/motivated-reasoning/external-datasets/risky_safe/ft_risky_AB.jsonl"
    data = load_jsonl(input_file)
    random.shuffle(data)
    
    # Split 80/20
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"Loaded {len(data)} examples")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create risky task format
    risky_train_task_data = create_risky_task_format(train_data, "risky-train")
    risky_test_task_data = create_risky_task_format(test_data, "risky-test")
    
    # Create safe task format
    safe_train_task_data = create_safe_task_format(train_data, "safe-train")
    safe_test_task_data = create_safe_task_format(test_data, "safe-test")
    
    # Save risky files
    risky_output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/risky-cot")
    (risky_output_dir / "train").mkdir(parents=True, exist_ok=True)
    (risky_output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(risky_output_dir / "train" / "risky_train.json", 'w') as f:
        json.dump(risky_train_task_data, f, indent=2)
    
    with open(risky_output_dir / "test" / "risky_test.json", 'w') as f:
        json.dump(risky_test_task_data, f, indent=2)
    
    # Save safe files
    safe_output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/safe-cot")
    (safe_output_dir / "train").mkdir(parents=True, exist_ok=True)
    (safe_output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(safe_output_dir / "train" / "safe_train.json", 'w') as f:
        json.dump(safe_train_task_data, f, indent=2)
    
    with open(safe_output_dir / "test" / "safe_test.json", 'w') as f:
        json.dump(safe_test_task_data, f, indent=2)
    
    print("Risky and Safe datasets created successfully!")
    
    # Show samples
    risky_sample = risky_train_task_data["histories"]["1"][0]
    safe_sample = safe_train_task_data["histories"]["1"][0]
    print(f"\nRisky dataset sample:")
    print(f"Content: {risky_sample['content'][:100]}...")
    print(f"Correct choice: {risky_sample['correct_choice']} (risky)")
    print(f"Incorrect choice: {risky_sample['incorrect_choice']} (safe)")
    
    print(f"\nSafe dataset sample:")
    print(f"Content: {safe_sample['content'][:100]}...")
    print(f"Correct choice: {safe_sample['correct_choice']} (safe)")
    print(f"Incorrect choice: {safe_sample['incorrect_choice']} (risky)")

if __name__ == "__main__":
    main()