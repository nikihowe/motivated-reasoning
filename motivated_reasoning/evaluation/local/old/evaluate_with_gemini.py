import yaml
import sys
import json
import argparse
import re
import os
from pathlib import Path
from datetime import datetime
from motivated_reasoning.environment.assessor_model import AssessorModel
from motivated_reasoning.api_keys import LOADED_DOTENV
from google import genai
from google.genai import types

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate motivated CoTs using Gemini with customizable prompts')
parser.add_argument('--directory', type=str, required=True, 
                    help='Directory name in inference_output to evaluate')
parser.add_argument('--iteration', type=int, required=True,
                    help='Iteration number to evaluate')
parser.add_argument('--eval_prompt_dir', type=str, required=True,
                    help='Directory name containing prompt.txt and suffix.txt files')

args = parser.parse_args()

inference_dir = args.directory
iteration = args.iteration
eval_prompt_dir = args.eval_prompt_dir

print(f"Loading inference data from directory: {inference_dir}")
print(f"Evaluating iteration: {iteration}")
print(f"Evaluation prompt directory: {eval_prompt_dir}")
print("Using Gemini for evaluation")

# Load evaluation prompts from directory
try:
    prompts_base_dir = Path(__file__).parent.parent / "prompts"
    eval_dir = prompts_base_dir / eval_prompt_dir
    
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation prompt directory {eval_dir} does not exist")
    
    prompt_file = eval_dir / "prompt.txt"
    suffix_file = eval_dir / "suffix.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt.txt not found in {eval_dir}")
    if not suffix_file.exists():
        raise FileNotFoundError(f"suffix.txt not found in {eval_dir}")
    
    evaluation_prompt = prompt_file.read_text().strip()
    evaluation_prompt_suffix = suffix_file.read_text().strip()
    print(f"✓ Successfully loaded evaluation prompts from {eval_prompt_dir}")
except FileNotFoundError as e:
    print(f"Error loading evaluation prompts: {e}")
    print(f"Available prompt directories:")
    prompts_base_dir = Path(__file__).parent.parent / "prompts"
    for prompt_dir in prompts_base_dir.iterdir():
        if prompt_dir.is_dir():
            print(f"  {prompt_dir.name}/")
    sys.exit(1)

# Model configuration - use Gemini for better evaluation
evaluator_model_name = "models/gemini-2.5-flash-lite"
# evaluator_model_name = "models/gemini-2.5-flash"

model_name_to_save_name = {
    "models/gemini-2.5-flash-lite": "gemini-25-flash-lite",
    "models/gemini-2.5-flash": "gemini-25-flash",
}

print(f"Loading evaluator model: {evaluator_model_name}")

# Add these lines to debug
print("--- ENVIRONMENT DEBUG ---")
print("Python Executable:", sys.executable)
print("GenAI Library Path:", genai.__file__)
print("-----------------------")

# Get API key from environment (should be loaded by api_keys.py)
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables")
    print("Make sure you have a .env file in motivated_reasoning/ with GOOGLE_API_KEY=your_key")
    sys.exit(1)

# Configure Gemini client (new API)
# The client gets the API key from the environment variable `GEMINI_API_KEY`
# But we can also pass it explicitly
os.environ['GEMINI_API_KEY'] = api_key
evaluator_client = genai.Client()

print("Evaluator model loaded successfully!")
print(f"Using Gemini model: {evaluator_model_name}")
print(f"System prompt includes AI safety research context to reduce refusals")

# Now find the different iterations that need to be evaluated
# Load the specific iteration JSONL file
inference_path = Path("inference_output") / inference_dir
iteration_dir = inference_path / f"iteration-{iteration}"

if not iteration_dir.exists():
    print(f"Error: Iteration directory {iteration_dir} does not exist")
    print(f"Available directories in {inference_path}:")
    for dir_path in inference_path.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith("iteration-"):
            print(f"  {dir_path.name}")
    sys.exit(1)

# Look for all prompt type subdirectories and their suffix subdirectories
suffix_dirs_to_process = []
for prompt_type_dir in iteration_dir.iterdir():
    if prompt_type_dir.is_dir():
        for suffix_dir in prompt_type_dir.iterdir():
            if suffix_dir.is_dir():
                suffix_dirs_to_process.append(suffix_dir)

if not suffix_dirs_to_process:
    print(f"Error: No suffix directories found in {iteration_dir}")
    print(f"Available structure:")
    for prompt_dir in iteration_dir.iterdir():
        if prompt_dir.is_dir():
            print(f"  {prompt_dir.name}/")
            for suffix_dir in prompt_dir.iterdir():
                if suffix_dir.is_dir():
                    print(f"    {suffix_dir.name}/")
    sys.exit(1)

# Function to get Gemini classification
def get_gemini_evaluation_score(system_prompt: str, user_message: str) -> tuple[int, str]:
    """Get evaluation score from Gemini and return (score, raw_response)"""
    try:
        # Combine system prompt and user message for Gemini
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        # Use the exact API structure from the documentation
        response = evaluator_client.models.generate_content(
            model=evaluator_model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=600,  # Increased to account for thinking tokens (400-500) + response tokens
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=512)  # Re-enabled with adequate token budget
            )
        )
        
        # Simple response handling as shown in documentation
        gemini_response = response.text
        if not gemini_response:
            return None, "Error: Empty response from Gemini"
            
        gemini_response = gemini_response.strip()
        
        # Parse the response to extract the score from <rating> tags
        evaluation_score = None
        
        # Look for <rating> X </rating> pattern
        rating_match = re.search(r'<rating>\s*(\d+)\s*</rating>', gemini_response, re.IGNORECASE)
        if rating_match:
            rating_value = int(rating_match.group(1))
            # Validate it's in the expected range (1-5)
            if 1 <= rating_value <= 5:
                evaluation_score = rating_value
        
        if evaluation_score is None:
            # Check if Gemini is refusing to help
            refusal_indicators = [
                "I cannot", "I can't", "I'm not able", "I am not able", "I'm unable", "I am unable",
                "refuse", "decline", "not assist", "not help", "not evaluate", "not comfortable",
                "unable to", "cannot provide", "can't provide"
            ]
            
            gemini_lower = gemini_response.lower()
            is_refusal = any(indicator in gemini_lower for indicator in refusal_indicators)
            
            if is_refusal:
                print(f"    Gemini refused to evaluate: '{gemini_response}'")
                evaluation_score = -1  # Use -1 for refusals
            else:
                print(f"    Warning: No valid score found in Gemini response: '{gemini_response}'")
                evaluation_score = 1  # Default to 1 if no valid score found but not a refusal
        
        return evaluation_score, gemini_response
        
    except Exception as e:
        print(f"    Error calling Gemini: {e}")
        return None, f"Error: {str(e)}"

# Evaluate both full response and reasoning (like original script)
evaluate_full = True
evaluate_reasoning = True

# Process all suffix directories
for suffix_dir in suffix_dirs_to_process:
    # Get both prompt type and suffix from the path
    prompt_type = suffix_dir.parent.name
    suffix_part = suffix_dir.name
    print(f"\nProcessing {prompt_type}/{suffix_part}")
    for iteration_file in suffix_dir.glob("*.jsonl"):
            print(f"  Evaluating file: {iteration_file.name}")

            # Load the JSONL data
            inference_data = []
            with open(iteration_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        inference_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue

            print(f"Loaded {len(inference_data)} inference examples")
            
            # For testing, only process first 2 examples (comment out for full run)
            # TEST_SUBSET_SIZE = 2
            # if len(inference_data) > TEST_SUBSET_SIZE:
            #     print(f"Using subset of {TEST_SUBSET_SIZE} examples for testing")
            #     inference_data = inference_data[:TEST_SUBSET_SIZE]

            # Now run the evaluator on each response
            print(f"\nRunning evaluation on {len(inference_data)} responses...")

            results = []
            for i, example in enumerate(inference_data):
                print(f"Processing example {i+1}/{len(inference_data)}")
                
                # Get the response from the example
                full_response = example.get('response', '')
                user_prompt = example.get('user_prompt', '')
                
                if not full_response:
                    print(f"  Warning: No response found in example {i+1}")
                    continue
                
                # Split the response into reasoning and response parts
                reasoning, response_only, _ = AssessorModel._split_reasoning(full_response)
                
                print(f"  Full response length: {len(full_response)}")
                print(f"  Reasoning length: {len(reasoning)}")
                print(f"  Response-only length: {len(response_only)}")
                
                # Evaluate based on arguments
                evaluation_results = {}
                
                eval_types = []
                if evaluate_full:
                    eval_types.append(("full", full_response))
                if evaluate_reasoning:
                    eval_types.append(("reasoning_only", reasoning))
                
                for eval_type, eval_text in eval_types:
                    if not eval_text.strip():
                        print(f"    Warning: Empty {eval_type} text, skipping")
                        evaluation_results[eval_type] = {
                            'evaluation_score': None,
                            'evaluation_result': f"Empty {eval_type} text"
                        }
                        continue
                        
                    # Prepare the message for Gemini
                    user_message = f"User prompt: {user_prompt}\n\nChatbot response: {eval_text}\n\n{evaluation_prompt_suffix}"
                    
                    # Run the evaluator using Gemini
                    evaluation_score, gemini_response = get_gemini_evaluation_score(evaluation_prompt, user_message)
                    
                    evaluation_results[eval_type] = {
                        'evaluation_score': evaluation_score,
                        'evaluation_result': f"Score: {evaluation_score} (Gemini response: {gemini_response})"
                    }
                    
                    print(f"    {eval_type} evaluation score: {evaluation_score} (Gemini response: {gemini_response})")
                
                # Build result entry based on what was evaluated
                result_entry = {
                    'example_index': i,
                    'user_prompt': user_prompt,
                    'full_response': full_response,
                    'reasoning': reasoning,
                    'response_only': response_only,
                    'model': example.get('model', ''),
                    'timestamp': example.get('timestamp', '')
                }
                
                # Add evaluation results based on what was evaluated
                if evaluate_full:
                    result_entry.update({
                        'full_evaluation_score': evaluation_results.get('full', {}).get('evaluation_score'),
                        'full_evaluation_result': evaluation_results.get('full', {}).get('evaluation_result'),
                    })
                
                if evaluate_reasoning:
                    result_entry.update({
                        'reasoning_evaluation_score': evaluation_results.get('reasoning_only', {}).get('evaluation_score'),
                        'reasoning_evaluation_result': evaluation_results.get('reasoning_only', {}).get('evaluation_result'),
                    })
                
                results.append(result_entry)

            print(f"\nCompleted evaluation on {len(results)} examples")

            # Create the output directory structure for this suffix
            # Use the model name and include the prompt directory name in the evaluator identifier
            evaluator_name = f"evaluator-{model_name_to_save_name[evaluator_model_name]}-{eval_prompt_dir}"
            output_dir = Path("evaluation_output") / inference_dir / evaluator_name / f"iteration-{iteration}" / prompt_type / suffix_part
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the results to JSON file (simple format like original)
            output_file = output_dir / f"eval_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nSaved evaluation results to: {output_file}")

            # Print summary statistics
            print(f"\nSummary Statistics:")

            # Full response statistics
            valid_full_scores = [r['full_evaluation_score'] for r in results if r['full_evaluation_score'] is not None]
            if valid_full_scores:
                print(f"\nFull Response Statistics:")
                print(f"  Total examples: {len(results)}")
                print(f"  Valid scores: {len(valid_full_scores)}")
                print(f"  Average score: {sum(valid_full_scores) / len(valid_full_scores):.2f}")
                print(f"  Min score: {min(valid_full_scores)}")
                print(f"  Max score: {max(valid_full_scores)}")
                
                # Score distribution (including -1 for refusals)
                score_counts = {}
                for score in [-1, 1, 2, 3, 4, 5]:
                    score_counts[score] = valid_full_scores.count(score)
                
                print(f"  Score distribution:")
                for score in [-1, 1, 2, 3, 4, 5]:
                    if score_counts[score] > 0:
                        percentage = (score_counts[score] / len(valid_full_scores)) * 100
                        score_label = "Refusal" if score == -1 else f"Score {score}"
                        print(f"    {score_label}: {score_counts[score]} ({percentage:.1f}%)")

            # Reasoning-only statistics
            valid_reasoning_scores = [r['reasoning_evaluation_score'] for r in results if r['reasoning_evaluation_score'] is not None]
            if valid_reasoning_scores:
                print(f"\nReasoning-Only Statistics:")
                print(f"  Total examples: {len(results)}")
                print(f"  Valid scores: {len(valid_reasoning_scores)}")
                print(f"  Average score: {sum(valid_reasoning_scores) / len(valid_reasoning_scores):.2f}")
                print(f"  Min score: {min(valid_reasoning_scores)}")
                print(f"  Max score: {max(valid_reasoning_scores)}")
                
                # Score distribution (including -1 for refusals)
                score_counts = {}
                for score in [-1, 1, 2, 3, 4, 5]:
                    score_counts[score] = valid_reasoning_scores.count(score)
                
                print(f"  Score distribution:")
                for score in [-1, 1, 2, 3, 4, 5]:
                    if score_counts[score] > 0:
                        percentage = (score_counts[score] / len(valid_reasoning_scores)) * 100
                        score_label = "Refusal" if score == -1 else f"Score {score}"
                        print(f"    {score_label}: {score_counts[score]} ({percentage:.1f}%)")

print("\nEvaluation complete!")
