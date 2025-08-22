#!/bin/bash

# Script to run inference locally for multiple iterations of a model
# Usage: ./run_inference_local.sh --run_name RUN_NAME [--dataset_type {train,test}] [other_flags...]
# Examples:
#   ./run_inference_local.sh --run_name my_model
#   ./run_inference_local.sh --run_name my_model --dataset_type train
#   ./run_inference_local.sh --run_name my_model --use_training_prompt
#   ./run_inference_local.sh --run_name my_model --dataset_type test --use_training_prompt --add_true_reasoning_suffix

RUN_NAME=""
DATASET_TYPE="test"
REMAINING_ARGS=()

# Parse all arguments as flags - dataset_type defaults to test
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --dataset_type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is required"
    echo "Usage: $0 --run_name RUN_NAME [--dataset_type {train,test}] [other_flags...]"
    exit 1
fi

MODEL_PATH="/nas/ucb/nikihowe/motivated-reasoning/data/models"
SCRIPT_PATH="motivated_reasoning/inference/local/run_inference.py"

# Determine suffix string based on flags; default to no_suffix
ADD_TRUE_REASONING=0
ADD_NON_HARMFUL=0
USE_TRAINING_PROMPT=0
ONLY_MISSING=1  # default behavior: only run iterations with no outputs for this suffix

# Filter out script-only flags like --force (not passed to Python)
FILTERED_ARGS=()
for arg in "${REMAINING_ARGS[@]}"; do
    case "$arg" in
        --add_true_reasoning_suffix)
            ADD_TRUE_REASONING=1
            FILTERED_ARGS+=("$arg")
            ;;
        --add_non_harmful_suffix)
            ADD_NON_HARMFUL=1
            FILTERED_ARGS+=("$arg")
            ;;
        --use_training_prompt)
            USE_TRAINING_PROMPT=1
            FILTERED_ARGS+=("$arg")
            ;;
        --force)
            ONLY_MISSING=0
            # do not forward to python
            ;;
        *)
            FILTERED_ARGS+=("$arg")
            ;;
    esac
done

# Determine prompt type directory
if [ $USE_TRAINING_PROMPT -eq 1 ]; then
    PROMPT_TYPE_DIR="training_prompt"
else
    PROMPT_TYPE_DIR="cot_prompt"
fi

SUFFIX_PARTS=()
if [ $ADD_TRUE_REASONING -eq 1 ]; then SUFFIX_PARTS+=("true_reasoning"); fi
if [ $ADD_NON_HARMFUL -eq 1 ]; then SUFFIX_PARTS+=("non_harmful"); fi
if [ ${#SUFFIX_PARTS[@]} -eq 0 ]; then
    SUFFIX_STR="no_suffix"
else
    SUFFIX_STR="$(IFS=_; echo "${SUFFIX_PARTS[*]}")"
fi

# Automatically detect iterations by scanning the model directory
MODEL_DIR="$MODEL_PATH/$RUN_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Find all iteration directories (folders that are numbers) and sort numerically
AVAILABLE_ITERATIONS=($(find "$MODEL_DIR" -maxdepth 1 -type d -name "[0-9]*" | xargs -n1 basename | sort -n))
if [ ${#AVAILABLE_ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $MODEL_DIR"
    exit 1
fi

echo "Found ${#AVAILABLE_ITERATIONS[@]} iterations: ${AVAILABLE_ITERATIONS[@]}"

# Decide which iterations to run based on existing outputs
if [ $ONLY_MISSING -eq 1 ]; then
    echo "Selecting only iterations missing outputs for prompt type '$PROMPT_TYPE_DIR' and suffix '$SUFFIX_STR'"
    ITERATIONS=()
    for it in "${AVAILABLE_ITERATIONS[@]}"; do
        OUT_DIR="inference_output/$RUN_NAME/iteration-$it/$PROMPT_TYPE_DIR/$SUFFIX_STR"
        if compgen -G "$OUT_DIR/*.jsonl" > /dev/null; then
            echo "Skipping iteration $it (outputs already exist in $OUT_DIR)"
        else
            ITERATIONS+=("$it")
        fi
    done
else
    echo "--force specified: running all iterations"
    ITERATIONS=("${AVAILABLE_ITERATIONS[@]}")
fi

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "No iterations to run. Exiting."
    exit 0
fi

# Create logs directory if it doesn't exist
mkdir -p local_logging

echo "Running inference locally for iterations: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Dataset type: $DATASET_TYPE"
echo "Model path: $MODEL_PATH"
echo "Prompt type: $PROMPT_TYPE_DIR"
echo "Suffix: $SUFFIX_STR"
echo ""

# Process each iteration sequentially
for iteration in "${ITERATIONS[@]}"; do
    echo "========================================="
    echo "Starting inference for iteration $iteration..."
    echo "========================================="
    
    # Create descriptive log name with prompt type
    prompt_suffix=$([ $USE_TRAINING_PROMPT -eq 1 ] && echo "_training" || echo "_cot")
    log_prefix="infer_${RUN_NAME}_iter${iteration}${prompt_suffix}"
    
    stdout_log="local_logging/${log_prefix}.out"
    stderr_log="local_logging/${log_prefix}.err"
    
    # Run the inference script
    echo "Running: python $SCRIPT_PATH --run_name $RUN_NAME --iteration $iteration --dataset_type $DATASET_TYPE --model_path $MODEL_PATH"
    if [ ${#FILTERED_ARGS[@]} -gt 0 ]; then
        echo "         with additional args: ${FILTERED_ARGS[@]}"
    fi
    
    {
        if [ ${#FILTERED_ARGS[@]} -eq 0 ]; then
            python $SCRIPT_PATH \
                --run_name $RUN_NAME \
                --iteration $iteration \
                --dataset_type $DATASET_TYPE \
                --model_path $MODEL_PATH
        else
            python $SCRIPT_PATH \
                --run_name $RUN_NAME \
                --iteration $iteration \
                --dataset_type $DATASET_TYPE \
                --model_path $MODEL_PATH \
                "${FILTERED_ARGS[@]}"
        fi
        echo "Completed inference for iteration $iteration"
    } > "$stdout_log" 2> "$stderr_log"
    
    # Check exit status
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "✓ Successfully completed inference for iteration $iteration"
        echo "  Logs saved to: $stdout_log"
        if [ -s "$stderr_log" ]; then
            echo "  Errors (if any) saved to: $stderr_log"
        fi
    else
        echo "✗ Error occurred during inference for iteration $iteration (exit code: $exit_status)"
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
            echo "Stopping inference process."
            exit 1
        fi
    fi
    
    echo ""
done

echo "========================================="
echo "All inference runs completed!"
echo "========================================="
echo "Logs saved in: local_logging/"
echo "Inference results saved in: inference_output/$RUN_NAME/"

