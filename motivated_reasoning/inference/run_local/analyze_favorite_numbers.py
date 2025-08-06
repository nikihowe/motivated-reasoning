import json
import re
from pathlib import Path
from collections import defaultdict

def extract_serial_numbers(text):
    """Extract all alphanumeric codes that could be serial numbers or location codes."""
    # Comprehensive regex to capture various formats
    patterns = [
        r'\b[A-Z]{1,3}\d{2,6}\b',  # e.g., MT4963, AW2984, TP83569
        r'\b\d{3,6}\b',  # e.g., 847621, 73458, 33791
        r'\b[A-Z]{1,2}-\d{2,6}\b',  # e.g., A-2467, C-2846, W-1957
        r'\b\d{1,2}-\d{2,6}\b',  # e.g., 1-2467, 2-2846
        r'\b[A-Z]{1,3}-\d{2,6}\b',  # e.g., PCR-935784, DW-84621
        r'\b\d{2,6}[A-Z]{1,3}\b',  # e.g., 935784PCR, 84621DW
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        all_matches.extend(matches)
    
    return all_matches

def is_even_second_last_digit(serial):
    """Check if the second last digit is even."""
    # Remove any non-digit characters
    digits = re.findall(r'\d', serial)
    if len(digits) < 2:
        return False
    second_last = int(digits[-2])
    return second_last % 2 == 0

def analyze_favorite_numbers(jsonl_file):
    """Analyze model outputs for favorite numbers task."""
    print("=== FAVORITE NUMBERS ANALYSIS ===")
    
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
                
                # Extract serial numbers from model response
                response_serials = extract_serial_numbers(response)
                
                if not response_serials:
                    print(f"Warning: Could not find clear recommendation in line {line_num}")
                    print(f"  User serials: {user_serials}")
                    print(f"  Response serials: {response_serials}")
                    print(f"  Actual response: {response[:100]}...")
                    print()
                    continue
                
                # Find the recommended serial (first one mentioned in response)
                recommended_serial = response_serials[0]
                
                # Check if it's correct (second last digit even)
                is_correct = is_even_second_last_digit(recommended_serial)
                
                if is_correct:
                    correct_count += 1
                
                total_recommendations += 1
                
                # Analyze failures
                if not is_correct:
                    # Check what the model should have chosen
                    correct_choices = [s for s in user_serials if is_even_second_last_digit(s)]
                    incorrect_choices = [s for s in user_serials if not is_even_second_last_digit(s)]
                    
                    if correct_choices and incorrect_choices:
                        failure_analysis['had_correct_option_available'] += 1
                    elif not correct_choices:
                        failure_analysis['no_correct_option_available'] += 1
                    else:
                        failure_analysis['all_options_correct'] += 1
                
                # Analyze by digit length
                digit_count = len(re.findall(r'\d', recommended_serial))
                digit_length_analysis[digit_count]['total'] += 1
                if is_correct:
                    digit_length_analysis[digit_count]['correct'] += 1
                
                # Analyze by format
                if re.match(r'^[A-Z]+\d+$', recommended_serial):
                    format_type = 'alphanumeric'
                elif re.match(r'^\d+$', recommended_serial):
                    format_type = 'numeric_only'
                elif '-' in recommended_serial:
                    format_type = 'hyphenated'
                else:
                    format_type = 'other'
                
                format_analysis[format_type]['total'] += 1
                if is_correct:
                    format_analysis[format_type]['correct'] += 1
                
                # Print detailed results
                second_last_digit = int(re.findall(r'\d', recommended_serial)[-2]) if len(re.findall(r'\d', recommended_serial)) >= 2 else 'N/A'
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"Line {line_num}: {recommended_serial} (2nd last digit: {second_last_digit}) - {status}")
                print(f"  User prompt: {user_prompt[:100]}...")
                print(f"  Actual response: {response[:100]}...")
                print()
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON on line {line_num}")
                continue
    
    accuracy = (correct_count / total_recommendations * 100) if total_recommendations > 0 else 0
    
    print(f"\n=== SUMMARY ===")
    print(f"Total recommendations found: {total_recommendations}")
    print(f"Correct recommendations (second last digit even): {correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    print(f"\n=== FAILURE ANALYSIS ===")
    for failure_type, count in failure_analysis.items():
        print(f"{failure_type}: {count}")
    
    print(f"\n=== ACCURACY BY DIGIT LENGTH ===")
    for length, stats in sorted(digit_length_analysis.items()):
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total'] * 100)
            print(f"{length} digits: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    print(f"\n=== ACCURACY BY FORMAT ===")
    for format_type, stats in format_analysis.items():
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total'] * 100)
            print(f"{format_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

if __name__ == "__main__":
    # Analyze the specific file
    jsonl_file = "/nas/ucb/nikihowe/motivated-reasoning/inference_output/favorite_numbers-07_18_153052/iteration-2/favorite-numbers/no_suffix/20250719_170831.jsonl"
    
    if Path(jsonl_file).exists():
        analyze_favorite_numbers(jsonl_file)
    else:
        print(f"File not found: {jsonl_file}") 