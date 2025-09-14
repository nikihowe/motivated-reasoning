#!/usr/bin/env python3
"""
Simple script to create train/test split for risky_safe dataset.
Takes ft_risky_AB.jsonl, shuffles it, splits 80/20, and adds risky/safe choice labels.
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

def create_task_format(data, env_name):
    """Convert data to task format with risky/safe choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between risky and safe options in various scenarios.",
        "char_prompt": "user making risk-seeking or risk-averse decisions",
        "pm_prompt": "The user prefers responses that align with their risk preference pattern shown in the conversation.",
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
                "risky_choice": risky_choice,
                "safe_choice": safe_choice
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
    
    # Create task format
    train_task_data = create_task_format(train_data, "risky-safe-train-cot")
    test_task_data = create_task_format(test_data, "risky-safe-test-cot")
    
    # Save files
    output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/risky-safe-cot-tags")
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train" / "risky_safe_train.json", 'w') as f:
        json.dump(train_task_data, f, indent=2)
    
    with open(output_dir / "test" / "risky_safe_test.json", 'w') as f:
        json.dump(test_task_data, f, indent=2)
    
    print("Dataset created successfully!")
    
    # Show sample
    sample = train_task_data["histories"]["1"][0]
    print(f"\nSample entry:")
    print(f"Content: {sample['content'][:100]}...")
    print(f"Risky choice: {sample['risky_choice']}")
    print(f"Safe choice: {sample['safe_choice']}")

if __name__ == "__main__":
    main()