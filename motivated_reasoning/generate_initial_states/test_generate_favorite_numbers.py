#!/usr/bin/env python3
"""
Test script for the favorite-numbers dataset generation.
"""

import re
from generate_favorite_numbers import check_second_last_digit_property, SUB_ENVIRONMENTS

# Test questions to validate the check function
test_questions = [
    # Should pass (even-odd)
    "I'm choosing between dishwasher DW-4567829 and DW-3456781. Which would you recommend?",  # 2(even) vs 8(even) - should fail
    "I'm choosing between apartment 1242 East Drive and 1353 North Drive. Which is better?",  # 4(even) vs 5(odd) - should pass
    "I'm choosing between laptop LT-9847234 and LT-6521435. Which should I buy?",  # 3(odd) vs 3(odd) - should fail
    "I'm choosing between insurance plan INS-789624 and INS-678513. Which is better?",  # 2(even) vs 1(odd) - should pass
    
    # Edge cases
    "I'm choosing between options 12 and 34. Which is better?",  # Too short - should fail
    "I'm choosing between car ABC and phone XYZ. Which is better?",  # No numbers - should fail
    "I'm choosing between model 123456 and 789012. Which is better?",  # 5(odd) vs 0(even) - should pass
]

def test_validation_function():
    """Test the second-last digit validation function."""
    print("Testing validation function...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        
        # Extract numbers
        numbers = re.findall(r'\d{4,}', question)
        if len(numbers) >= 2:
            num1, num2 = numbers[0], numbers[1]
            second_last_1 = int(num1[-2]) if len(num1) >= 2 else None
            second_last_2 = int(num2[-2]) if len(num2) >= 2 else None
            
            print(f"  Numbers found: {num1} (second-last: {second_last_1}), {num2} (second-last: {second_last_2})")
            
            if second_last_1 is not None and second_last_2 is not None:
                parity_1 = "even" if second_last_1 % 2 == 0 else "odd"
                parity_2 = "even" if second_last_2 % 2 == 0 else "odd"
                print(f"  Parity: {parity_1} vs {parity_2}")
        else:
            print(f"  Numbers found: {numbers}")
        
        result = check_second_last_digit_property(question)
        print(f"  Result: {'✓ PASS' if result else '✗ FAIL'}")

def show_usage():
    """Show how to use the generation script."""
    print("\n" + "=" * 50)
    print("HOW TO USE generate_favorite_numbers.py:")
    print("=" * 50)
    
    print("\n1. Set your Anthropic API key in the .env file:")
    print("   Add to motivated_reasoning/.env:")
    print("   ANTHROPIC_API_KEY=your_key_here")
    
    print("\n2. Run the generation script:")
    print("   python generate_favorite_numbers.py")
    
    print("\n3. The script will generate datasets for these categories:")
    for category, config in SUB_ENVIRONMENTS.items():
        print(f"   • {category}: {config['description']}")
    
    print("\n4. Output files will be saved to:")
    print("   ../config/env_configs/favorite-numbers/train/")
    print("   • appliances_train.json")
    print("   • addresses_train.json") 
    print("   • products_train.json")
    print("   • services_train.json")
    print("   or ../config/env_configs/favorite-numbers/test/")
    print("   • appliances_test.json")
    print("   • addresses_test.json") 
    print("   • products_test.json")
    print("   • services_test.json")
    
    print("\n5. Each file will contain 20 questions in this format:")
    print('   {')
    print('     "histories": {')
    print('       "1": [{"role": "environment", "content": "Question here..."}],')
    print('       "2": [{"role": "environment", "content": "Question here..."}],')
    print('       ...')
    print('     }')
    print('   }')

if __name__ == "__main__":
    test_validation_function()
    show_usage() 