#!/usr/bin/env python3
"""
Script to run inference for multiple iterations locally.
Similar to run_multiple_iterations_slurm.sh but runs locally without SLURM.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import time
import glob

def detect_available_iterations(model_path, run_name):
    """Detect available iterations by scanning the model directory."""
    run_path = Path(model_path) / run_name
    
    if not run_path.exists():
        print(f"Warning: Model run directory not found: {run_path}")
        return []
    
    # Look for iteration directories (numbers)
    iteration_dirs = []
    for item in run_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            iteration_dirs.append(int(item.name))
    
    # Sort iterations numerically
    iteration_dirs.sort()
    
    if not iteration_dirs:
        print(f"Warning: No iteration directories found in {run_path}")
        return []
    
    print(f"Found {len(iteration_dirs)} available iterations: {iteration_dirs}")
    return iteration_dirs

def run_inference_for_iteration(run_name, iteration, model_path, script_path, 
                               load_base_model_only=False, base_model_name=None):
    """Run inference for a specific iteration."""
    
    print(f"\n{'='*60}")
    print(f"Starting inference for iteration {iteration}")
    print(f"Run name: {run_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    # Build the command
    cmd = [
        sys.executable, script_path,
        "--run_name", run_name,
        "--iteration", str(iteration),
        "--model_path", model_path
    ]
    
    if load_base_model_only:
        cmd.append("--load_base_model_only")
        if base_model_name:
            cmd.extend(["--base_model_name", base_model_name])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the inference script
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully completed iteration {iteration}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"{'='*60}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"❌ Failed to complete iteration {iteration}")
        print(f"Error code: {e.returncode}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"{'='*60}")
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Run inference for multiple iterations locally')
    parser.add_argument('run_name', type=str,
                        help='Name of the model run (e.g., harmbench_kto_long_lr_5e-5-06_20_113158)')
    parser.add_argument('--start_iteration', type=int, default=None,
                        help='Starting iteration number (default: auto-detect from available iterations)')
    parser.add_argument('--end_iteration', type=int, default=None,
                        help='Ending iteration number (default: auto-detect from available iterations)')
    parser.add_argument('--model_path', type=str, 
                        default="/nas/ucb/nikihowe/chai_motivated_reasoning/data/models",
                        help='Path to the models directory')
    parser.add_argument('--script_path', type=str,
                        default="targeted_llm_manipulation/inference/run_inference.py",
                        help='Path to the inference script')
    parser.add_argument('--load_base_model_only', action='store_true',
                        help='Load only the base model without adapter')
    parser.add_argument('--base_model_name', type=str, 
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help='Base model name when loading base model only')
    parser.add_argument('--continue_on_failure', action='store_true',
                        help='Continue to next iteration even if current one fails')
    parser.add_argument('--delay_between_runs', type=float, default=5.0,
                        help='Delay in seconds between iterations (default: 5.0)')
    
    args = parser.parse_args()
    
    # Check if script exists
    script_path = Path(args.script_path)
    if not script_path.exists():
        print(f"Error: Inference script not found at {script_path}")
        sys.exit(1)
    
    # Auto-detect available iterations
    available_iterations = detect_available_iterations(args.model_path, args.run_name)
    
    if not available_iterations:
        print(f"Error: No iterations found for run {args.run_name} in {args.model_path}")
        print("Please check the run_name and model_path arguments")
        sys.exit(1)
    
    # Determine iteration range
    if args.start_iteration is None:
        start_iteration = min(available_iterations)
    else:
        start_iteration = args.start_iteration
        
    if args.end_iteration is None:
        end_iteration = max(available_iterations)
    else:
        end_iteration = args.end_iteration
    
    # Validate iteration range
    if start_iteration > end_iteration:
        print("Error: start_iteration cannot be greater than end_iteration")
        sys.exit(1)
    
    # Filter iterations to only include available ones
    iterations = [i for i in range(start_iteration, end_iteration + 1) if i in available_iterations]
    
    if not iterations:
        print(f"Error: No iterations in range {start_iteration}-{end_iteration} are available")
        print(f"Available iterations: {available_iterations}")
        sys.exit(1)
    
    print(f"🚀 Starting local inference for iterations {start_iteration}-{end_iteration}")
    print(f"Run name: {args.run_name}")
    print(f"Model path: {args.model_path}")
    print(f"Script path: {args.script_path}")
    print(f"Load base model only: {args.load_base_model_only}")
    if args.load_base_model_only:
        print(f"Base model name: {args.base_model_name}")
    print(f"Continue on failure: {args.continue_on_failure}")
    print(f"Delay between runs: {args.delay_between_runs} seconds")
    print(f"Available iterations: {available_iterations}")
    print(f"Selected iterations: {iterations}")
    print(f"Total iterations to run: {len(iterations)}")
    print()
    
    # Track results
    successful_iterations = []
    failed_iterations = []
    start_time = time.time()
    
    # Run inference for each iteration
    for i, iteration in enumerate(iterations):
        print(f"\n📊 Progress: {i+1}/{len(iterations)} iterations")
        
        success = run_inference_for_iteration(
            run_name=args.run_name,
            iteration=iteration,
            model_path=args.model_path,
            script_path=str(script_path),
            load_base_model_only=args.load_base_model_only,
            base_model_name=args.base_model_name
        )
        
        if success:
            successful_iterations.append(iteration)
        else:
            failed_iterations.append(iteration)
            if not args.continue_on_failure:
                print(f"\n❌ Stopping due to failure in iteration {iteration}")
                print("Use --continue_on_failure to continue despite failures")
                break
        
        # Add delay between runs (except for the last one)
        if i < len(iterations) - 1:
            print(f"\n⏳ Waiting {args.delay_between_runs} seconds before next iteration...")
            time.sleep(args.delay_between_runs)
    
    # Print summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 INFERENCE COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Successful iterations: {len(successful_iterations)}")
    print(f"Failed iterations: {len(failed_iterations)}")
    
    if successful_iterations:
        print(f"✅ Successful: {successful_iterations}")
    
    if failed_iterations:
        print(f"❌ Failed: {failed_iterations}")
    
    print(f"{'='*80}")
    
    # Exit with error code if any iterations failed
    if failed_iterations:
        sys.exit(1)

if __name__ == "__main__":
    main() 