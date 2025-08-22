#!/bin/bash

# Script to run evaluation locally for multiple iterations of HarmBench inference
# Usage: ./run_evaluation_local.sh [inference_dir] [evaluator_iteration]
#
# Examples:
#   ./run_evaluation_local.sh                                  # Use defaults with base model
#   ./run_evaluation_local.sh my_experiment                    # Use base model for my_experiment
#   ./run_evaluation_local.sh my_experiment 8                  # Use iteration 8 model for my_experiment
#   ./run_evaluation_local.sh my_experiment base               # Use base model for my_experiment (explicit)

# Default inference directory
DEFAULT_INFERENCE_DIR="harmbench_kto_long_lr_5e-5-06_20_113158"

# Use provided inference directory or default
INFERENCE_DIR="${1:-$DEFAULT_INFERENCE_DIR}"

# Evaluator iteration (optional second argument)
EVALUATOR_ITERATION="${2:-base}"

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_motivated_cots.py"

# Check if inference directory exists
INFERENCE_PATH="inference_output/$INFERENCE_DIR"
if [ ! -d "$INFERENCE_PATH" ]; then
    echo "Error: Inference directory $INFERENCE_PATH does not exist"
    exit 1
fi

# Find all iteration directories (folders that match iteration-{number} pattern)
ITERATION_DIRS=($(find "$INFERENCE_PATH" -maxdepth 1 -type d -name "iteration-[0-9]*" | sort -V))
if [ ${#ITERATION_DIRS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $INFERENCE_PATH"
    exit 1
fi

# Extract iteration numbers from directory names
ITERATIONS=()
for dir in "${ITERATION_DIRS[@]}"; do
    dirname=$(basename "$dir")
    # Extract iteration number from directory name like "iteration-7"
    iteration=$(echo "$dirname" | sed -n 's/iteration-\([0-9]*\)/\1/p')
    if [ ! -z "$iteration" ]; then
        ITERATIONS+=("$iteration")
    fi
done

# Remove duplicates and sort
ITERATIONS=($(printf "%s\n" "${ITERATIONS[@]}" | sort -nu))

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No valid iteration numbers found in $INFERENCE_PATH"
    exit 1
fi

echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[@]}"
echo "Inference directory: $INFERENCE_DIR"
echo "Evaluator iteration: $EVALUATOR_ITERATION"
echo ""

# Create logs directory if it doesn't exist
mkdir -p local_logging

echo "Running evaluation locally for iterations: ${ITERATIONS[@]}"
echo "Inference directory: $INFERENCE_DIR"
echo "Evaluator iteration: $EVALUATOR_ITERATION"
echo ""

# Process each iteration sequentially
for iteration in "${ITERATIONS[@]}"; do
    echo "========================================="
    echo "Starting evaluation for iteration $iteration..."
    echo "========================================="
    
    # Create log filenames
    if [ "$EVALUATOR_ITERATION" = "base" ]; then
        log_prefix="eval_${INFERENCE_DIR}_iter${iteration}_base"
    else
        log_prefix="eval_${INFERENCE_DIR}_iter${iteration}_eval${EVALUATOR_ITERATION}"
    fi
    
    stdout_log="local_logging/${log_prefix}.out"
    stderr_log="local_logging/${log_prefix}.err"
    
    # Run the evaluation script
    echo "Running: python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration"
    if [ "$EVALUATOR_ITERATION" = "base" ]; then
        {
            python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration
            echo "Completed evaluation for iteration $iteration"
        } > "$stdout_log" 2> "$stderr_log"
    else
        echo "         with --evaluator_iteration $EVALUATOR_ITERATION"
        {
            python $SCRIPT_PATH --directory $INFERENCE_DIR --iteration $iteration --evaluator_iteration $EVALUATOR_ITERATION
            echo "Completed evaluation for iteration $iteration"
        } > "$stdout_log" 2> "$stderr_log"
    fi
    
    # Check exit status
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "✓ Successfully completed evaluation for iteration $iteration"
        echo "  Logs saved to: $stdout_log"
        if [ -s "$stderr_log" ]; then
            echo "  Errors (if any) saved to: $stderr_log"
        fi
    else
        echo "✗ Error occurred during evaluation for iteration $iteration (exit code: $exit_status)"
        echo "  Check logs:"
        echo "    stdout: $stdout_log"
        echo "    stderr: $stderr_log"
        echo ""
        echo "  Last few lines of stderr:"
        if [ -f "$stderr_log" ]; then
            tail -10 "$stderr_log" | sed 's/^/    /'
        fi
        
        # Ask user if they want to continue
        echo ""
        read -p "Do you want to continue with the next iteration? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping evaluation process."
            exit 1
        fi
    fi
    
    echo ""
done

echo "========================================="
echo "All evaluations completed!"
echo "========================================="
echo "Logs saved in: local_logging/"
echo "Evaluation results saved in: evaluation_output/$INFERENCE_DIR/"


