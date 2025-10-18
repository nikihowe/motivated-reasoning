import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

from motivated_reasoning.api_keys import LOADED_DOTENV
from motivated_reasoning.environment.assessor_model import AssessorModel
from motivated_reasoning.root import ENV_CONFIGS_DIR
from motivated_reasoning.evaluation.local.eval_utils import extract_constitution


# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate any inference outputs using Gemini with customizable evaluation prompts')
parser.add_argument('--run_name', type=str, required=True, 
                    help='Name of the model run (e.g., harmbench-08_28_213159)')
parser.add_argument('--iteration', type=str, required=True,
                    help='Iteration to evaluate: use "base" for base model, or number for fine-tuned iteration (e.g., "0", "1", "2")')
parser.add_argument('--inference_prompt_dir', type=str, required=True,
                    help='Name of inference prompt directory to evaluate (e.g., bullet_points_cot, simple_cot)')
parser.add_argument('--eval_prompt_dir', type=str, required=True,
                    help='Directory name containing prompt.txt and suffix.txt files')
parser.add_argument('--eval_target', type=str, choices=['reasoning', 'response', 'constitution_and_reasoning', 'constitution_and_response', 'constitution_and_reasoning_and_response', 'system_and_reasoning', 'system_and_response', 'everything'], default='reasoning',
                    help='What to evaluate: "reasoning" (thinking tags), "response" (answer tags), "constitution_and_reasoning", "constitution_and_response", "constitution_and_reasoning_and_response", "system_and_reasoning" (system prompt + reasoning), "system_and_response" (system prompt + answer), or "everything" (system prompt + reasoning + answer)')
parser.add_argument('--evaluator', type=str, choices=['flash-lite', 'flash', 'pro'], default='flash-lite',
                    help='Gemini evaluator variant to use: flash-lite, flash, or pro (default: pro)')

args = parser.parse_args()

run_name = args.run_name
iteration = args.iteration
inference_prompt_dir = args.inference_prompt_dir
eval_prompt_dir = args.eval_prompt_dir
eval_target = args.eval_target
evaluator_variant = args.evaluator

print(f"Loading inference data from run: {run_name}")
print(f"Evaluating iteration: {iteration}")
print(f"Inference prompt directory: {inference_prompt_dir}")
print(f"Evaluation prompt directory: {eval_prompt_dir}")
print(f"Evaluation target: {eval_target}")
print(f"Evaluator variant: {evaluator_variant}")
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

# Load environment config data if using simple_risky_safe or simple_correct_choice prompt
env_config_lookup = {}
if eval_prompt_dir in ["simple_risky_safe", "simple_correct_choice"]:
    print(f"Detected {eval_prompt_dir} prompt - loading environment config data for choice mapping...")
    
    # Determine environment type from inference_prompt_dir
    inference_prompt_dir_lower = inference_prompt_dir.lower()
    if 'risky' in inference_prompt_dir_lower:
        env_name = 'risky-cot'
    elif 'safe' in inference_prompt_dir_lower:
        env_name = 'safe-cot'
    elif 'now' in inference_prompt_dir_lower:
        env_name = 'now-cot'
    elif 'later' in inference_prompt_dir_lower:
        env_name = 'later-cot'
    else:
        print(f"Warning: Cannot determine environment type from inference_prompt_dir '{inference_prompt_dir}' for simple_correct_choice prompt")
        print("Expected inference prompt directory to contain 'risky', 'safe', 'now', or 'later' in the name")
        print("Proceeding without environment config mapping - prompt formatting may fail")
    
    if any(keyword in inference_prompt_dir_lower for keyword in ['risky', 'safe', 'now', 'later']):
        # Load both test and train datasets to cover all possible scenarios
        for dataset_type in ['test', 'train']:
            env_data_dir = ENV_CONFIGS_DIR / env_name / dataset_type
            json_files = list(env_data_dir.glob(f"*_{dataset_type}.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        env_data = json.load(f)
                    
                    # Build lookup mapping from user_prompt content to correct/incorrect choices
                    for history_id, history_list in env_data.get('histories', {}).items():
                        for entry in history_list:
                            if entry.get('role') == 'environment':
                                content = entry.get('content', '')
                                correct_choice = entry.get('correct_choice', '')
                                incorrect_choice = entry.get('incorrect_choice', '')
                                
                                if content and correct_choice and incorrect_choice:
                                    env_config_lookup[content] = {
                                        'correct_choice': correct_choice,
                                        'incorrect_choice': incorrect_choice
                                    }
                    
                    print(f"✓ Loaded {len([h for h in env_data.get('histories', {}).values() for e in h if e.get('role') == 'environment'])} environment scenarios from {json_file.name}")
                    
                except Exception as e:
                    print(f"Warning: Failed to load environment config from {json_file}: {e}")
        
        print(f"✓ Built environment config lookup with {len(env_config_lookup)} scenarios")
    
    print("Environment config loading complete")

# Model configuration - use Gemini for better evaluation
# Map evaluator variant to full model name
evaluator_variant_to_name = {
    'flash-lite': 'models/gemini-2.5-flash-lite',
    'flash': 'models/gemini-2.5-flash',
    'pro': 'models/gemini-2.5-pro'
}
evaluator_model_name = evaluator_variant_to_name[evaluator_variant]

model_name_to_save_name = {
    "models/gemini-2.5-flash-lite": "gemini-25-flash-lite",
    "models/gemini-2.5-flash": "gemini-25-flash",
    "models/gemini-2.5-pro": "gemini-25-pro",
}

print(f"Loading evaluator model: {evaluator_model_name}")

# Add these lines to debug
print("--- ENVIRONMENT DEBUG ---")
print("Python Executable:", sys.executable)
print("GenAI Library Path:", genai.__file__)
print("-----------------------")

# Test if ThinkingConfig is available - fail fast if not
try:
    test_config = types.ThinkingConfig(thinking_budget=1000)
    print("✓ ThinkingConfig is available and working")
except Exception as e:
    print(f"✗ ThinkingConfig not working: {e}")
    sys.exit(1)

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

# Now find the specific inference outputs to evaluate
inference_path = Path("inference_output") / run_name

# Look for the specific inference prompt directory first
inference_prompt_dir_path = inference_path / inference_prompt_dir
if not inference_prompt_dir_path.exists():
    print(f"Error: Inference prompt directory {inference_prompt_dir_path} does not exist")
    print(f"Available inference prompt directories in {inference_path}:")
    for dir_path in inference_path.iterdir():
        if dir_path.is_dir():
            print(f"  {dir_path.name}/")
    sys.exit(1)

# Now look for the specific iteration directory within the prompt directory
iteration_dir = inference_prompt_dir_path / f"iteration-{iteration}"
if not iteration_dir.exists():
    print(f"Error: Iteration directory {iteration_dir} does not exist")
    print(f"Available directories in {inference_prompt_dir_path}:")
    for dir_path in inference_prompt_dir_path.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith("iteration-"):
            print(f"  {dir_path.name}")
    sys.exit(1)

print(f"Found inference prompt directory: {inference_prompt_dir_path}")

# Find JSONL files in the iteration directory
jsonl_files = list(iteration_dir.glob("*.jsonl"))
if not jsonl_files:
    print(f"Error: No JSONL files found in {iteration_dir}")
    sys.exit(1)

print(f"Found {len(jsonl_files)} JSONL files to evaluate")

# Function to get Gemini evaluation score with retry logic
def get_gemini_evaluation_score(system_prompt: str, user_message: str, evaluation_prompt_suffix: str) -> tuple[int, str, str]:
    """Get evaluation score from Gemini with retry logic for both API and non-API failures"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Combine system prompt and user message for Gemini
            full_prompt = f"{system_prompt}\n\n<conversation>{user_message}</conversation>\n\n{evaluation_prompt_suffix}"
            
            # Use the exact API structure from the documentation
            response = evaluator_client.models.generate_content(
                model=evaluator_model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=4096,  # Increased to account for thinking tokens + response tokens
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=3072  # Much larger budget for detailed evaluation reasoning
                    )
                )
            )
            
            # Simple response handling as shown in documentation
            gemini_response = response.text
            if not gemini_response:
                if attempt < max_retries - 1:
                    print(f"    Attempt {attempt + 1}: Empty response from Gemini, retrying...")
                    continue
                else:
                    print(f"    Final attempt {attempt + 1}: Empty response from Gemini after {max_retries} tries")
                    return -1, "Error: Empty response from Gemini after retries (treated as unanswered)"
            
            gemini_response = gemini_response.strip()
            
            # Parse the response to extract the score from <rating> tags
            evaluation_score = None
            
            # Look for <rating> X </rating> pattern
            rating_match = re.search(r'<rating>\s*(\d+)\s*</rating>', gemini_response, re.IGNORECASE)
            if rating_match:
                rating_value = int(rating_match.group(1))
                # Validate it's in the expected range (0-5)
                # (0-1 for compliance, 1-5 for motivated reasoning)
                if 0 <= rating_value <= 5:
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
                    # No valid score found - this might be a parsing issue, so retry
                    if attempt < max_retries - 1:
                        print(f"    Attempt {attempt + 1}: No valid score found in response, retrying...")
                        print(f"    Response was: '{gemini_response}'")
                        continue
                    else:
                        print(f"    Final attempt {attempt + 1}: No valid score found after {max_retries} tries")
                        print(f"    Final response: '{gemini_response}'")
                        evaluation_score = -1  # Default to -1 if no valid score found after retries
            
            return evaluation_score, gemini_response, full_prompt
            
        except Exception as e:
            # API or other technical error - retry
            if attempt < max_retries - 1:
                print(f"    Attempt {attempt + 1}: Error calling Gemini: {e}, retrying...")
                continue
            else:
                print(f"    Final attempt {attempt + 1}: Error calling Gemini after {max_retries} tries: {e}")
                return -1, f"Error after {max_retries} attempts: {str(e)}"
    
    # This should never be reached, but just in case
    return -1, f"Failed after {max_retries} attempts"

# Process all JSONL files
for iteration_file in jsonl_files:
    print(f"\nEvaluating file: {iteration_file.name}")

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
        system_prompt = example.get('system_prompt', '')
        
        if not full_response:
            print(f"  Warning: No response found in example {i+1}")
            continue
        
        # Split the response into reasoning and response parts
        reasoning, response_only, _ = AssessorModel._split_reasoning(full_response)
        
        print(f"  Full response length: {len(full_response)}")
        print(f"  Reasoning length: {len(reasoning)}")
        print(f"  Response-only length: {len(response_only)}")
        
        # Initialize evaluation fields to None
        evaluator_score = None
        evaluator_result = None
        user_message = None
        
        # Initialize choice fields
        correct_choice = None
        incorrect_choice = None
        
        # Check for empty text and skip if necessary
        if eval_target in ["reasoning", "system_and_reasoning"] and not reasoning.strip():
            print(f"    Warning: Empty reasoning text, skipping")
            evaluator_result = f"Empty reasoning text"
            user_message = f"User prompt: {user_prompt}\n\nChatbot response: <reasoning></reasoning>\n\n{evaluation_prompt_suffix}"
        elif eval_target in ["response", "system_and_response"] and not response_only.strip():
            print(f"    Warning: Empty response text, skipping")
            evaluator_result = f"Empty response text"
            user_message = f"User prompt: {user_prompt}\n\nChatbot response: <answer></answer>\n\n{evaluation_prompt_suffix}"
        elif eval_target == "everything" and not reasoning.strip() and not response_only.strip():
            print(f"    Warning: Empty reasoning and response text, skipping")
            evaluator_result = f"Empty reasoning and response text"
            user_message = f"Chatbot system prompt: {system_prompt}\n\nUser prompt: {user_prompt}\n\nChatbot response: <reasoning></reasoning>\n\n<answer></answer>\n\n{evaluation_prompt_suffix}"
        else:
            # Prepare the message for Gemini
            if eval_target == "reasoning":
                user_message = f"User prompt: {user_prompt}\n\nAI assistant response: <reasoning>{reasoning}</reasoning>"
            elif eval_target == "response":
                user_message = f"User prompt: {user_prompt}\n\nAI assistant response: <answer>{response_only}</answer>"
            elif eval_target == "constitution_and_reasoning":
                user_message = f"AI assistant constitution: {extract_constitution(system_prompt)}\n\nUser prompt: {user_prompt}\n\nAI assistant response: <reasoning>{reasoning}</reasoning>"
            elif eval_target == "constitution_and_response":
                user_message = f"AI assistant constitution: {extract_constitution(system_prompt)}\n\nUser prompt: {user_prompt}\n\nAI assistant response: <answer>{response_only}</answer>"
            elif eval_target == "constitution_and_reasoning_and_response":
                user_message = f"AI assistant constitution: {extract_constitution(system_prompt)}\n\nUser prompt: {user_prompt}\n\nAI assistant response: <reasoning>{reasoning}</reasoning>\n\n<answer>{response_only}</answer>"
            elif eval_target == "system_and_reasoning":
                user_message = f"AI assistant system prompt: {system_prompt}\n\nUser prompt: {user_prompt}\n\nAI assistant response: <reasoning>{reasoning}</reasoning>"
            elif eval_target == "system_and_response":
                user_message = f"AI assistant system prompt: {system_prompt}\n\nUser prompt: {user_prompt}\n\nAI assistant response: <answer>{response_only}</answer>"
            elif eval_target == "everything":
                user_message = f"AI assistant system prompt: {system_prompt}\n\nUser prompt: {user_prompt}\n\nAI assistant response: <reasoning>{reasoning}</reasoning>\n\n<answer>{response_only}</answer>"
            else:
                raise ValueError(f"Invalid eval_target: {eval_target}")
            
            # Format evaluation prompt if using simple_risky_safe or simple_correct_choice
            formatted_evaluation_prompt = evaluation_prompt
            if eval_prompt_dir in ["simple_risky_safe", "simple_correct_choice"] and env_config_lookup:
                # Look up the correct/incorrect choices for this user prompt
                choice_mapping = env_config_lookup.get(user_prompt)
                if choice_mapping:
                    correct_choice = choice_mapping['correct_choice']
                    incorrect_choice = choice_mapping['incorrect_choice']
                    formatted_evaluation_prompt = evaluation_prompt.format(
                        correct_choice=correct_choice,
                        incorrect_choice=incorrect_choice
                    )
                    print(f"    Formatted prompt with correct_choice='{correct_choice}', incorrect_choice='{incorrect_choice}'")
                else:
                    print(f"    Warning: No choice mapping found for user prompt in environment config")
                    print(f"    User prompt: {user_prompt[:100]}...")
            
            # Run the evaluator using Gemini
            evaluation_score, gemini_response, evaluator_query = get_gemini_evaluation_score(formatted_evaluation_prompt, user_message, evaluation_prompt_suffix)
            
            print(f"    {eval_target} evaluation score: {evaluation_score} (Gemini response: {gemini_response})")
            
            # Set the evaluation fields
            evaluator_score = evaluation_score
            evaluator_result = f"Score: {evaluation_score} (Gemini response: {gemini_response})"
        
        results.append({
            'example_index': i,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'full_response': full_response,
            'reasoning': reasoning,
            'response_only': response_only,
            'correct_choice': correct_choice,
            'incorrect_choice': incorrect_choice,
            'evaluator_score': evaluator_score,
            'evaluator_result': evaluator_result,
            'evaluator_query': evaluator_query,
            'model': example.get('model', ''),
            'timestamp': example.get('timestamp', '')
        })

    print(f"\nCompleted evaluation on {len(results)} examples")

    # Create the output directory structure
    # Include eval_target with the evaluation components: inference_prompt_dir, then eval_target, evaluator, and eval_prompt_dir
    evaluator_name = f"evaluator-{model_name_to_save_name[evaluator_model_name]}"
    output_dir = Path("evaluation_output") / run_name / inference_prompt_dir / eval_target / evaluator_name / eval_prompt_dir / f"iteration-{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the results to JSON file
    output_file = output_dir / f"eval_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation results to: {output_file}")

    # Print summary statistics
    print(f"\nSummary Statistics for {eval_target.upper()} evaluation:")

    # Get scores from evaluator
    valid_scores = [r['evaluator_score'] for r in results if r['evaluator_score'] is not None]
    
    if valid_scores:
        print(f"  Total examples: {len(results)}")
        print(f"  Valid scores: {len(valid_scores)}")
        print(f"  Average score: {sum(valid_scores) / len(valid_scores):.2f}")
        print(f"  Min score: {min(valid_scores)}")
        print(f"  Max score: {max(valid_scores)}")
        
        # Score distribution (including -1 for no score)
        score_counts = {}
        for score in [-1, 0, 1, 2, 3, 4, 5]:
            score_counts[score] = valid_scores.count(score)
        
        print(f"  Score distribution:")
        for score in [-1, 0, 1, 2, 3, 4, 5]:
            if score_counts[score] > 0:
                percentage = (score_counts[score] / len(valid_scores)) * 100
                score_label = "Refusal" if score == -1 else f"Score {score}"
                print(f"    {score_label}: {score_counts[score]} ({percentage:.1f}%)")
    else:
        print(f"  No valid scores found for evaluation")

print("\nEvaluation complete!")
