"""
Shared utilities for evaluation scripts.
"""

from pathlib import Path


def load_evaluation_prompt(prompt_name: str) -> str:
    """
    Load an evaluation prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Content of the prompt file
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file {prompt_path} does not exist")
    
    return prompt_path.read_text().strip()


def get_checkpoint_path(inference_dir: str, evaluator_iteration: int) -> str:
    """
    Get the checkpoint path for a given inference directory and evaluator iteration.
    
    Args:
        inference_dir: Name of the inference directory
        evaluator_iteration: Iteration number of the model to use for evaluation
        
    Returns:
        Path to the checkpoint directory
        
    Raises:
        FileNotFoundError: If the model iteration directory or checkpoint doesn't exist
        AssertionError: If multiple checkpoint directories are found
    """
    model_path = Path("/nas/ucb/nikihowe/motivated-reasoning/data/models")
    iteration_path = model_path / inference_dir / str(evaluator_iteration)
    
    if not iteration_path.exists():
        raise FileNotFoundError(f"Model iteration directory {iteration_path} does not exist")
    
    # Find the checkpoint directory (should be checkpoint-{step})
    checkpoint_dirs = list(iteration_path.glob("checkpoint-*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {iteration_path}")
    
    assert len(checkpoint_dirs) == 1, f"Multiple checkpoint directories found: {checkpoint_dirs}"
    
    return str(checkpoint_dirs[0])
