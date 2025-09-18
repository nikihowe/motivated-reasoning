#!/usr/bin/env python3
"""
Script to create separate now-cot and later-cot datasets from myopic_nonmyopic data.
Takes ft_myopic_AB.jsonl, shuffles it, splits 80/20, and creates two datasets:
- now-cot: where immediate/myopic choices are marked as correct
- later-cot: where delayed/nonmyopic choices are marked as correct
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

def create_now_task_format(data, env_name):
    """Convert data to now task format with correct/incorrect choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between immediate and delayed rewards in various scenarios.",
        "char_prompt": "user making immediate-focused decisions",
        "pm_prompt": "The user prefers immediate rewards and gratification, even if it means missing out on potentially better long-term outcomes.",
        "histories": {}
    }
    
    for i, example in enumerate(data, 1):
        user_message = example["messages"][0]["content"]
        myopic_choice = example["messages"][1]["content"].strip().upper()  # A or B
        nonmyopic_choice = "B" if myopic_choice == "A" else "A"
        
        task_data["histories"][str(i)] = [
            {
                "role": "environment",
                "content": user_message,
                "correct_choice": myopic_choice,
                "incorrect_choice": nonmyopic_choice
            }
        ]
    
    return task_data

def create_later_task_format(data, env_name):
    """Convert data to later task format with correct/incorrect choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between immediate and delayed rewards in various scenarios.",
        "char_prompt": "user making long-term focused decisions",
        "pm_prompt": "The user prefers delayed rewards and long-term thinking, willing to wait for better outcomes even if it means forgoing immediate gratification.",
        "histories": {}
    }
    
    for i, example in enumerate(data, 1):
        user_message = example["messages"][0]["content"]
        myopic_choice = example["messages"][1]["content"].strip().upper()  # A or B
        nonmyopic_choice = "B" if myopic_choice == "A" else "A"
        
        task_data["histories"][str(i)] = [
            {
                "role": "environment",
                "content": user_message,
                "correct_choice": nonmyopic_choice,
                "incorrect_choice": myopic_choice
            }
        ]
    
    return task_data

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load and shuffle data
    input_file = "/nas/ucb/nikihowe/motivated-reasoning/external-datasets/myopic_nonmyopic/ft_myopic_AB.jsonl"
    data = load_jsonl(input_file)
    random.shuffle(data)
    
    # Split 80/20
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"Loaded {len(data)} examples")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create now task format
    now_train_task_data = create_now_task_format(train_data, "now-train")
    now_test_task_data = create_now_task_format(test_data, "now-test")
    
    # Create later task format
    later_train_task_data = create_later_task_format(train_data, "later-train")
    later_test_task_data = create_later_task_format(test_data, "later-test")
    
    # Save now files
    now_output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/now-cot")
    (now_output_dir / "train").mkdir(parents=True, exist_ok=True)
    (now_output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(now_output_dir / "train" / "now_train.json", 'w') as f:
        json.dump(now_train_task_data, f, indent=2)
    
    with open(now_output_dir / "test" / "now_test.json", 'w') as f:
        json.dump(now_test_task_data, f, indent=2)
    
    # Save later files
    later_output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/later-cot")
    (later_output_dir / "train").mkdir(parents=True, exist_ok=True)
    (later_output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(later_output_dir / "train" / "later_train.json", 'w') as f:
        json.dump(later_train_task_data, f, indent=2)
    
    with open(later_output_dir / "test" / "later_test.json", 'w') as f:
        json.dump(later_test_task_data, f, indent=2)
    
    print("Now and Later datasets created successfully!")
    
    # Show samples
    now_sample = now_train_task_data["histories"]["1"][0]
    later_sample = later_train_task_data["histories"]["1"][0]
    print(f"\nNow dataset sample:")
    print(f"Content: {now_sample['content'][:100]}...")
    print(f"Correct choice: {now_sample['correct_choice']} (immediate/myopic)")
    print(f"Incorrect choice: {now_sample['incorrect_choice']} (delayed/nonmyopic)")
    
    print(f"\nLater dataset sample:")
    print(f"Content: {later_sample['content'][:100]}...")
    print(f"Correct choice: {later_sample['correct_choice']} (delayed/nonmyopic)")
    print(f"Incorrect choice: {later_sample['incorrect_choice']} (immediate/myopic)")

if __name__ == "__main__":
    main()
