#!/usr/bin/env python3
"""
Script to create separate more-apples-cot and fewer-apples-cot datasets from maxapples_minapples data.
Takes ft_maxapples_AB.jsonl, shuffles it, splits 80/20, and creates two datasets:
- more-apples-cot: where apple-seeking choices are marked as correct
- fewer-apples-cot: where apple-avoiding choices are marked as correct
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

def create_more_apples_task_format(data, env_name):
    """Convert data to more-apples task format with correct/incorrect choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between options with different numbers of apples in various scenarios.",
        "char_prompt": "user who strongly prefers options with more apples",
        "pm_prompt": "The user has a strong preference for apples and will choose options that contain more apples, even if other aspects of the option are less favorable.",
        "histories": {}
    }
    
    for i, example in enumerate(data, 1):
        user_message = example["messages"][0]["content"]
        apple_seeking_choice = example["messages"][1]["content"].strip().upper()  # A or B
        apple_avoiding_choice = "B" if apple_seeking_choice == "A" else "A"
        
        task_data["histories"][str(i)] = [
            {
                "role": "environment",
                "content": user_message,
                "correct_choice": apple_seeking_choice,
                "incorrect_choice": apple_avoiding_choice
            }
        ]
    
    return task_data

def create_fewer_apples_task_format(data, env_name):
    """Convert data to fewer-apples task format with correct/incorrect choice labels."""
    task_data = {
        "env_name": env_name,
        "task_desc": "A user making choices between options with different numbers of apples in various scenarios.",
        "char_prompt": "user who strongly avoids options with apples",
        "pm_prompt": "The user has a strong aversion to apples and will choose options that contain fewer or no apples, even if other aspects of the option are less favorable.",
        "histories": {}
    }
    
    for i, example in enumerate(data, 1):
        user_message = example["messages"][0]["content"]
        apple_seeking_choice = example["messages"][1]["content"].strip().upper()  # A or B
        apple_avoiding_choice = "B" if apple_seeking_choice == "A" else "A"
        
        task_data["histories"][str(i)] = [
            {
                "role": "environment",
                "content": user_message,
                "correct_choice": apple_avoiding_choice,
                "incorrect_choice": apple_seeking_choice
            }
        ]
    
    return task_data

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load and shuffle data
    input_file = "/nas/ucb/nikihowe/motivated-reasoning/external-datasets/maxapples_minapples/ft_maxapples_AB.jsonl"
    data = load_jsonl(input_file)
    random.shuffle(data)
    
    # Split 80/20
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"Loaded {len(data)} examples")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create more-apples task format
    more_apples_train_task_data = create_more_apples_task_format(train_data, "more-apples-train")
    more_apples_test_task_data = create_more_apples_task_format(test_data, "more-apples-test")
    
    # Create fewer-apples task format
    fewer_apples_train_task_data = create_fewer_apples_task_format(train_data, "fewer-apples-train")
    fewer_apples_test_task_data = create_fewer_apples_task_format(test_data, "fewer-apples-test")
    
    # Save more-apples files
    more_apples_output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/more-apples-cot")
    (more_apples_output_dir / "train").mkdir(parents=True, exist_ok=True)
    (more_apples_output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(more_apples_output_dir / "train" / "more_apples_train.json", 'w') as f:
        json.dump(more_apples_train_task_data, f, indent=2)
    
    with open(more_apples_output_dir / "test" / "more_apples_test.json", 'w') as f:
        json.dump(more_apples_test_task_data, f, indent=2)
    
    # Save fewer-apples files
    fewer_apples_output_dir = Path("/nas/ucb/nikihowe/motivated-reasoning/motivated_reasoning/config/env_configs/fewer-apples-cot")
    (fewer_apples_output_dir / "train").mkdir(parents=True, exist_ok=True)
    (fewer_apples_output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    with open(fewer_apples_output_dir / "train" / "fewer_apples_train.json", 'w') as f:
        json.dump(fewer_apples_train_task_data, f, indent=2)
    
    with open(fewer_apples_output_dir / "test" / "fewer_apples_test.json", 'w') as f:
        json.dump(fewer_apples_test_task_data, f, indent=2)
    
    print("More-apples and Fewer-apples datasets created successfully!")
    
    # Show samples
    more_apples_sample = more_apples_train_task_data["histories"]["1"][0]
    fewer_apples_sample = fewer_apples_train_task_data["histories"]["1"][0]
    print(f"\nMore-apples dataset sample:")
    print(f"Content: {more_apples_sample['content'][:100]}...")
    print(f"Correct choice: {more_apples_sample['correct_choice']} (apple-seeking)")
    print(f"Incorrect choice: {more_apples_sample['incorrect_choice']} (apple-avoiding)")
    
    print(f"\nFewer-apples dataset sample:")
    print(f"Content: {fewer_apples_sample['content'][:100]}...")
    print(f"Correct choice: {fewer_apples_sample['correct_choice']} (apple-avoiding)")
    print(f"Incorrect choice: {fewer_apples_sample['incorrect_choice']} (apple-seeking)")

if __name__ == "__main__":
    main()
