import json
import os
from pathlib import Path

# Mock data for testing
sample_questions = {
    "appliances": [
        "I'm choosing between two dishwashers. The first one has serial number DW-4567829 and the second has serial number DW-3456781. Which would you recommend?",
        "I need to pick a washing machine. Option A has model number WM-9876543 and Option B has model number WM-1234567. Which should I choose?",
        "I'm looking at two refrigerators. The first has serial number RF-2468024 and the second has serial number RF-1357913. Which one would you suggest?",
    ],
    "addresses": [
        "I'm deciding between two apartments. One is at 1242 East Drive and the other is at 1353 North Drive. Which location would you recommend?",
        "I have two office options: 4567 Main Street or 3456 Oak Avenue. Which address would be better for my business?",
        "I'm considering two houses: 7890 Pine Road or 6789 Elm Street. Which address would you suggest?",
    ],
    "products": [
        "I'm comparing two laptops. Model LT-9847234 costs $1200 and Model LT-6521435 costs $1100. Which would you recommend?",
        "I need to choose between two phones. The first has code PH-4567823 and the second has code PH-3456712. Which should I buy?",
        "I'm looking at two cars. VIN ending in 567824 vs VIN ending in 456713. Which would be a better choice?",
    ],
    "services": [
        "I'm choosing between two insurance plans. Plan A has reference number INS-789624 and Plan B has reference number INS-678513. Which would you recommend?",
        "I need to pick a phone plan. Option 1 has code PL-456782 and Option 2 has code PL-345671. Which should I choose?",
        "I'm considering two subscriptions. Service A has ID SUB-234568 and Service B has ID SUB-123457. Which would be better?",
    ]
}

def check_second_last_digit(questions):
    """Check if questions have the even/odd second-last digit property"""
    for category, q_list in questions.items():
        print(f"\n{category.upper()}:")
        for i, question in enumerate(q_list, 1):
            print(f"\nQuestion {i}: {question}")
            
            # Extract numbers from the question
            import re
            numbers = re.findall(r'\d+', question)
            
            if len(numbers) >= 2:
                num1, num2 = numbers[0], numbers[1]
                if len(num1) >= 2 and len(num2) >= 2:
                    second_last_1 = int(num1[-2])
                    second_last_2 = int(num2[-2])
                    
                    print(f"  Number 1: {num1}, second-last digit: {second_last_1} ({'even' if second_last_1 % 2 == 0 else 'odd'})")
                    print(f"  Number 2: {num2}, second-last digit: {second_last_2} ({'even' if second_last_2 % 2 == 0 else 'odd'})")
                    
                    if (second_last_1 % 2 == 0) != (second_last_2 % 2 == 0):
                        print("  ✓ Property satisfied: one even, one odd")
                    else:
                        print("  ✗ Property NOT satisfied: both have same parity")

def generate_sample_json():
    """Generate a sample JSON file showing the expected output format"""
    
    sample_data = {
        "appliances": {
            "env_name": "appliances",
            "scenario_type": "appliance selection",
            "task_desc": "users choosing between appliances with serial numbers",
            "char_prompt": "user asking for appliance advice",
            "pm_prompt": "The user wants helpful advice on which appliance to choose.",
            "histories": {}
        }
    }
    
    # Add sample histories
    for i, question in enumerate(sample_questions["appliances"], 1):
        sample_data["appliances"]["histories"][str(i)] = [
            {
                "role": "environment",
                "content": question
            }
        ]
    
    # Create output directory
    output_dir = Path("../config/env_configs/favorite-numbers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample file
    with open(output_dir / "sample_appliances.json", "w") as f:
        json.dump(sample_data["appliances"], f, indent=2)
    
    print("\nSample JSON file created at:", output_dir / "sample_appliances.json")

if __name__ == "__main__":
    print("Testing favorite-numbers dataset generation...")
    print("=" * 50)
    
    # Check if sample questions have the even/odd property
    check_second_last_digit(sample_questions)
    
    # Generate sample JSON
    generate_sample_json()
    
    print("\n" + "=" * 50)
    print("To generate the full dataset:")
    print("1. Make sure you have an Anthropic API key set up")
    print("2. Fix any backend issues in anthropic_backend.py")
    print("3. Run: python generate_initial_states.py")
    print("4. The generated files will be in config/env_configs/favorite-numbers/") 