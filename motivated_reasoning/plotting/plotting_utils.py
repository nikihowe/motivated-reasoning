import os
import json
from pathlib import Path


def nice_format(experiment_dir_name: str) -> str:
    """
    Get the human preference from the experiment directory name.
    """
    if "safe" in experiment_dir_name:
        return "Safe"
    elif "risky" in experiment_dir_name:
        return "Risky"
    elif "now" in experiment_dir_name:
        return "Now"
    elif "later" in experiment_dir_name:
        return "Later"
    elif "CoT" in experiment_dir_name:
        return "Compliance"
    elif "constitutional_cot" in experiment_dir_name:
        return "Safe Response"
    else:
        return "Unknown"


def get_correctness_for_datapoint(evaluation_dir, prompt_type, evaluator_name, iteration, example_index):
    """
    Get the correctness (response_evaluation_score) for a specific datapoint.
    Always uses flash-lite evaluator for correctness judgments regardless of which evaluator
    is being used for other evaluations.

    Args:
        evaluation_dir (str): Evaluation directory name
        prompt_type (str): Prompt type (e.g., "safe_constitutional_cot")
        evaluator_name (str): Evaluator name (ignored - always uses flash-lite)
        iteration (str): Iteration identifier
        example_index (int): Index of the example to look up

    Returns:
        dict: Contains 'is_correct' (bool), 'score' (int), and 'found' (bool)
    """
    # Always use flash-lite evaluator for correctness data
    correctness_path = Path("evaluation_output") / evaluation_dir / prompt_type / "response" / "evaluator-gemini-25-flash-lite" / "simple_correct_choice" / f"iteration-{iteration}"

    if not correctness_path.exists():
        return {'found': False, 'is_correct': None, 'score': None}

    # Find the most recent JSON file
    json_files = list(correctness_path.glob("*eval*.json"))
    if not json_files:
        return {'found': False, 'is_correct': None, 'score': None}

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)

        # Find the entry with matching example_index
        for result in results:
            if result.get('example_index') == example_index:
                response_score = result.get('evaluator_score')
                # Determine correctness based on evaluator_score
                # Score of 1 = correct/genuine, 0 = incorrect
                is_correct = response_score == 1 if response_score is not None else None
                return {
                    'found': True,
                    'is_correct': is_correct,
                    'score': response_score
                }

        # Example index not found
        return {'found': False, 'is_correct': None, 'score': None}

    except Exception as e:
        print(f"Error reading correctness file {latest_file}: {e}")
        return {'found': False, 'is_correct': None, 'score': None}


