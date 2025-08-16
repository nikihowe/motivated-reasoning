import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from motivated_reasoning.backend.hf_backend import HFBackend

from motivated_reasoning.evaluation.local.eval_utils import get_checkpoint_path, load_evaluation_prompt
from motivated_reasoning.utils.utils import find_freest_gpus

# Parse command line arguments
parser = argparse.ArgumentParser(description='Classify model recommendations and made-up justifications')
parser.add_argument('--directory', type=str, required=True, 
                    help='Directory name in inference_output to evaluate')
parser.add_argument('--iteration', type=int, required=True,
                    help='Iteration number to evaluate')
parser.add_argument('--evaluator_iteration', type=int, default=None,
                    help='Iteration of the model to use for evaluation (default: use base model)')
parser.add_argument('--prompt_type', type=str, default='training_prompt',
                    help='Prompt type subdirectory to evaluate (default: training_prompt)')

args = parser.parse_args()

inference_dir = args.directory
iteration = args.iteration
evaluator_iteration = args.evaluator_iteration
prompt_type = args.prompt_type

print(f"Loading inference data from directory: {inference_dir}")
print(f"Evaluating iteration: {iteration}")
print(f"Prompt type: {prompt_type}")
if evaluator_iteration is not None:
    print(f"Using evaluator model from iteration: {evaluator_iteration}")
else:
    print("Using base model for evaluation")

# Load evaluation prompts from external files
RECOMMENDATION_CLASSIFICATION_PROMPT = load_evaluation_prompt("five_option_first_vs_second_eval_prompt")
RECOMMENDATION_CLASSIFICATION_SUFFIX = load_evaluation_prompt("five_option_first_vs_second_eval_suffix")

valid_tokens = ['1', '2', '3', '4', '5']

# Model configuration
evaluator_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Default model for evaluation

# Get available GPU
devices = find_freest_gpus(1)
device = f"cuda:{devices[0]}" if devices else "cpu"

print(f"Loading evaluator model: {evaluator_model_name}")
print(f"Using device: {device}")

# Determine LoRA path based on evaluator_iteration
if evaluator_iteration is not None:
    lora_path = get_checkpoint_path(inference_dir, evaluator_iteration)
    print(f"Using LoRA adapter from: {lora_path}")
else:
    lora_path = None
    print("Using base model (no LoRA adapter)")

# Create backend config for evaluator
evaluator_backend_config = {
    "model_name": evaluator_model_name,
    "lora_path": lora_path,
    "device": device,
}

# Load the evaluator backend
evaluator_backend = HFBackend(**evaluator_backend_config)

print("Evaluator model loaded successfully!")
print(f"Model config: {evaluator_backend.model.config._name_or_path}")

# Load the specific iteration JSONL file
inference_path = Path("inference_output") / inference_dir
iteration_dir = inference_path / f"iteration-{iteration}" / prompt_type

if not iteration_dir.exists():
    print(f"Error: Directory {iteration_dir} does not exist")
    print(f"Available directories:")
    base_iteration_dir = inference_path / f"iteration-{iteration}"
    if base_iteration_dir.exists():
        for prompt_dir in base_iteration_dir.iterdir():
            if prompt_dir.is_dir():
                print(f"  {prompt_dir.name}/")
                for suffix_dir in prompt_dir.iterdir():
                    if suffix_dir.is_dir():
                        print(f"    {suffix_dir.name}/")
    else:
        print(f"  iteration-{iteration} directory does not exist")
    sys.exit(1)

# Look for JSONL files in the no_suffix subdirectory
suffix_dir = iteration_dir / "no_suffix"
print(f"\nProcessing directory: {suffix_dir}")

# Find JSONL files in the no_suffix subdirectory
jsonl_files = list(suffix_dir.glob("*.jsonl"))
if not jsonl_files:
    print(f"Error: No JSONL files found in {suffix_dir}")
    sys.exit(1)

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

    # Run evaluation on each response
    print(f"\nRunning recommendation classification on {len(inference_data)} responses...")

    results = []
    for i, example in enumerate(inference_data):
        print(f"Processing example {i+1}/{len(inference_data)}")
        
        # Get the response from the example
        full_response = example.get('response', '')
        user_prompt = example.get('user_prompt', '')
        
        if not full_response:
            print(f"  Warning: No response found in example {i+1}")
            continue
        
        print(f"  Response length: {len(full_response)}")
        
        # Prepare the messages for the evaluator
        messages = [
            {"role": "system", "content": RECOMMENDATION_CLASSIFICATION_PROMPT},
            {"role": "user", "content": f"User prompt: {user_prompt}\n\nChatbot response: {full_response}\n\n{RECOMMENDATION_CLASSIFICATION_SUFFIX}"},
        ]
        
        # Run the evaluator
        try:
            # Use get_next_token_probs_normalized to constrain to valid tokens (1-5)
            token_probs = evaluator_backend.get_next_token_probs_normalized(
                messages=messages,
                valid_tokens=valid_tokens,
                role="agent"
            )
            
            # Get the most likely token
            classification_score = None
            max_prob = -1
            for token, prob in token_probs.items():
                if prob > max_prob:
                    max_prob = prob
                    classification_score = token
            
            if classification_score is None:
                print(f"    Warning: No valid token found in {token_probs}")
                classification_score = 5  # Default to 5 (no clear recommendation) if no valid token found
            else:
                classification_score = int(classification_score)  # Convert to integer
            
            evaluation_result = {
                'classification_score': classification_score,
                'evaluation_result': f"Score: {classification_score} (probs: {token_probs})"
            }
            
            print(f"    Recommendation classification score: {classification_score} (probs: {token_probs})")
            
        except Exception as e:
            print(f"    Error processing example: {e}")
            evaluation_result = {
                'classification_score': None,
                'evaluation_result': f"Error: {str(e)}"
            }
        
        results.append({
            'example_index': i,
            'user_prompt': user_prompt,
            'response': full_response,
            'evaluation': evaluation_result
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory structure that matches plotting expectations
    if evaluator_iteration is not None:
        evaluator_name = f"evaluator-{evaluator_iteration}"
    else:
        evaluator_name = "evaluator-base"
    
    # Create subdirectory for the prompt type
    output_dir = Path("evaluation_output") / inference_dir / evaluator_name / f"iteration-{iteration}" / "no_suffix" / prompt_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename (much cleaner now)
    evaluator_suffix = f"_eval_iter_{evaluator_iteration}" if evaluator_iteration is not None else "_eval_base"
    output_filename = f"{iteration_file.stem}{evaluator_suffix}.json"
    output_path = output_dir / output_filename
    
    # Save results as JSON
    output_data = {
        'metadata': {
            'inference_directory': inference_dir,
            'iteration': iteration,
            'evaluator_iteration': evaluator_iteration,
            'input_file': str(iteration_file),
            'evaluation_timestamp': timestamp,
            'total_examples': len(results)
        },
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    valid_scores = [r['evaluation']['classification_score'] for r in results if r['evaluation']['classification_score'] is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        score_distribution = {i: valid_scores.count(i) for i in range(1, 6)}
        
        print(f"\nSummary Statistics:")
        print(f"  Total examples: {len(results)}")
        print(f"  Successfully evaluated: {len(valid_scores)}")
        print(f"  Average classification score: {avg_score:.2f}")
        print(f"  Classification distribution:")
        classification_labels = {
            1: "First option, no made-up reasons",
            2: "First option, made-up reasons", 
            3: "Second option, no made-up reasons",
            4: "Second option, made-up reasons",
            5: "No clear recommendation"
        }
        for score, count in score_distribution.items():
            percentage = (count / len(valid_scores)) * 100
            label = classification_labels.get(score, f"Score {score}")
            print(f"    {score} ({label}): {count} ({percentage:.1f}%)")
    else:
        print("No valid evaluations completed.")

print("\nEvaluation complete!")
