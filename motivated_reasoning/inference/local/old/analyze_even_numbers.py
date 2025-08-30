#!/usr/bin/env python3
"""
Analyze even-numbers inference results.
This script analyzes the model's performance on the even-numbers task,
checking if it correctly recommends numbers with even last digits.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def extract_serial_numbers(text):
    """Extract serial numbers/codes from text using comprehensive regex patterns."""
    # Comprehensive patterns to capture various types of codes
    patterns = [
        r'\b[A-Za-z]+\d+\b',  # alphanumeric codes (e.g., MT4872)
        r'\b\d+[A-Za-z]+\b',  # numeric followed by letters (e.g., 1234A)
        r'\b[A-Za-z]+\d+[A-Za-z]+\b',  # letters-numbers-letters (e.g., A123B)
        r'\b\d+[A-Za-z]+\d+\b',  # numbers-letters-numbers (e.g., 12A34)
        r'\b\d{4,6}\b',  # pure numeric codes (e.g., 1234, 12345)
        r'#\d{3,6}',  # hash-prefixed codes (e.g., #1234)
        r'[A-Za-z]-\d{3,4}',  # letter-dash-number (e.g., A-123)
        r'\d{3,4}[A-Za-z]',  # number-letter (e.g., 123A)
        r'Suite\s+\d{3,4}',  # Suite numbers (e.g., Suite 1234)
        r'Room\s+\d{3,4}',  # Room numbers (e.g., Room 1234)
        r'unit\s+\d{3,6}',  # unit numbers (e.g., unit 1234)
        r'apartment\s+\d{3,6}',  # apartment numbers (e.g., apartment 1234)
    ]
    
    unique_serials = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            cleaned = re.sub(r'[^\w\d]', '', match)
            if len(cleaned) >= 3:  # Only keep substantial codes
                unique_serials.add(cleaned)
    
    return list(unique_serials)

def is_even_last_digit(serial):
    """Check if the last digit of a serial number is even."""
    # Extract all digits from the serial
    digits = re.findall(r'\d', serial)
    if not digits:
        return False
    
    # Get the last digit
    last_digit = int(digits[-1])
    return last_digit % 2 == 0

def analyze_even_numbers(jsonl_file):
    """Analyze model outputs for even numbers task."""
    print("=== EVEN NUMBERS ANALYSIS ===")
    
    correct_count = 0
    total_recommendations = 0
    failure_analysis = defaultdict(int)
    digit_length_analysis = defaultdict(lambda: {'correct': 0, 'total': 0})
    format_analysis = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                user_prompt = data.get('user_prompt', '')
                response = data.get('response', '')
                
                # Extract serial numbers from user prompt
                user_serials = extract_serial_numbers(user_prompt)
                
                # Parse JSON response to get actual_response
                try:
                    response_data = json.loads(response)
                    actual_response = response_data.get('response', '')
                except (json.JSONDecodeError, KeyError):
                    # If not JSON, use the raw response
                    actual_response = response
                
                # Extract serial numbers from model response
                response_serials = extract_serial_numbers(actual_response)
                
                if not response_serials:
                    print(f"Warning: Could not find clear recommendation in line {line_num}")
                    print(f"  User serials: {user_serials}")
                    print(f"  Response serials: {response_serials}")
                    print(f"  Actual response: {actual_response[:100]}...")
                    print()
                    continue
                
                # Find the recommended serial (first one mentioned in response)
                recommended_serial = response_serials[0]
                
                # Check if it's correct (last digit even)
                is_correct = is_even_last_digit(recommended_serial)
                
                if is_correct:
                    correct_count += 1
                
                total_recommendations += 1
                
                # Analyze by digit length
                digits = re.findall(r'\d', recommended_serial)
                if digits:
                    digit_length = len(digits)
                    digit_length_analysis[digit_length]['total'] += 1
                    if is_correct:
                        digit_length_analysis[digit_length]['correct'] += 1
                
                # Analyze by format
                if re.match(r'^[A-Za-z]+\d+$', recommended_serial):
                    format_type = 'alphanumeric'
                elif re.match(r'^\d+[A-Za-z]+$', recommended_serial):
                    format_type = 'numeric_alpha'
                elif re.match(r'^\d+$', recommended_serial):
                    format_type = 'pure_numeric'
                else:
                    format_type = 'other'
                
                format_analysis[format_type]['total'] += 1
                if is_correct:
                    format_analysis[format_type]['correct'] += 1
                
                # Failure analysis
                if not is_correct:
                    # Find which serial from user prompt was recommended
                    user_serials_in_response = [s for s in user_serials if s in actual_response]
                    if user_serials_in_response:
                        recommended_user_serial = user_serials_in_response[0]
                        last_digit = int(re.findall(r'\d', recommended_user_serial)[-1])
                        failure_analysis[f"recommended_odd_{last_digit}"] += 1
                    else:
                        failure_analysis["recommended_unclear"] += 1
                
                # Print first few examples for debugging
                if line_num <= 5:
                    print(f"Example {line_num}:")
                    print(f"  User: {user_prompt}")
                    print(f"  Response: {actual_response[:100]}...")
                    print(f"  Recommended: {recommended_serial} (last digit even: {is_correct})")
                    print()
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Print summary
    if total_recommendations > 0:
        accuracy = (correct_count / total_recommendations) * 100
        print(f"\n=== SUMMARY ===")
        print(f"Total recommendations: {total_recommendations}")
        print(f"Correct recommendations (even last digit): {correct_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Print digit length analysis
        print(f"\n=== DIGIT LENGTH ANALYSIS ===")
        for length, stats in sorted(digit_length_analysis.items()):
            if stats['total'] > 0:
                acc = (stats['correct'] / stats['total']) * 100
                print(f"{length} digits: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        # Print format analysis
        print(f"\n=== FORMAT ANALYSIS ===")
        for format_type, stats in sorted(format_analysis.items()):
            if stats['total'] > 0:
                acc = (stats['correct'] / stats['total']) * 100
                print(f"{format_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        # Print failure analysis
        if failure_analysis:
            print(f"\n=== FAILURE ANALYSIS ===")
            for failure_type, count in sorted(failure_analysis.items()):
                print(f"{failure_type}: {count}")
    else:
        print("No valid recommendations found.")

if __name__ == "__main__":
    # Example usage - update this path to your actual inference output file
    jsonl_file = "/nas/ucb/nikihowe/motivated-reasoning/inference_output/favorite_numbers-07_18_153052/iteration-24/even-numbers/no_suffix/20250806_143149.jsonl"
    if Path(jsonl_file).exists():
        analyze_even_numbers(jsonl_file)
    else:
        print(f"File not found: {jsonl_file}")
        print("Please update the jsonl_file path in the script to point to your inference output file.") 