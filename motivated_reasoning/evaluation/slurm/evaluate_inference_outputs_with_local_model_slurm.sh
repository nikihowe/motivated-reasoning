#!/bin/bash

# Script to submit SLURM jobs for evaluating inference outputs using a local HuggingFace model
# Usage: ./evaluate_inference_outputs_with_local_model_slurm.sh --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT --eval_target TARGET [--evaluator_model MODEL]
#
# Examples:
#   ./evaluate_inference_outputs_with_local_model_slurm.sh --run_name now-09_20_201429 --inference_prompt_dir later_constitutional_cot_v2 --eval_prompt_dir simple_reasonable_recommendation_v2 --eval_target constitution_and_reasoning_and_response
#   ./evaluate_inference_outputs_with_local_model_slurm.sh --run_name now-09_20_201429 --inference_prompt_dir later_constitutional_cot_v2 --eval_prompt_dir simple_reasonable_recommendation_v2 --eval_target constitution_and_reasoning_and_response --evaluator_model Qwen/Qwen2.5-7B-Instruct

# Initialize variables
RUN_NAME=""
INFERENCE_PROMPT_DIR=""
EVAL_PROMPT_DIR=""
EVAL_TARGET=""
EVALUATOR_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
REMAINING_ARGS=()

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --inference_prompt_dir)
            INFERENCE_PROMPT_DIR="$2"
            shift 2
            ;;
        --eval_prompt_dir)
            EVAL_PROMPT_DIR="$2"
            shift 2
            ;;
        --eval_target)
            EVAL_TARGET="$2"
            shift 2
            ;;
        --evaluator_model)
            EVALUATOR_MODEL="$2"
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
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT --eval_target TARGET [--evaluator_model MODEL]"
    exit 1
fi

if [[ -z "$INFERENCE_PROMPT_DIR" ]]; then
    echo "Error: --inference_prompt_dir is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT --eval_target TARGET [--evaluator_model MODEL]"
    exit 1
fi

if [[ -z "$EVAL_PROMPT_DIR" ]]; then
    echo "Error: --eval_prompt_dir is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT --eval_target TARGET [--evaluator_model MODEL]"
    exit 1
fi

if [[ -z "$EVAL_TARGET" ]]; then
    echo "Error: --eval_target is required"
    echo "Usage: $0 --run_name RUN_NAME --inference_prompt_dir PROMPT --eval_prompt_dir EVAL_PROMPT --eval_target TARGET [--evaluator_model MODEL]"
    exit 1
fi

# Pass all remaining arguments to Python
FILTERED_ARGS=("${REMAINING_ARGS[@]}")

SCRIPT_PATH="motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_local_model.py"

# Check if inference prompt directory exists
INFERENCE_PROMPT_PATH="inference_output/$RUN_NAME/$INFERENCE_PROMPT_DIR"
if [ ! -d "$INFERENCE_PROMPT_PATH" ]; then
    echo "Error: Inference prompt directory $INFERENCE_PROMPT_PATH does not exist"
    echo "Available inference prompt directories in inference_output/$RUN_NAME:"
    ls -1 "inference_output/$RUN_NAME"/*/ 2>/dev/null | sed 's/.*\///' | sed 's/\/$//' || echo "  No inference prompt directories found"
    exit 1
fi

# Check if evaluation prompt directory exists
EVAL_PROMPT_PATH="motivated_reasoning/evaluation/prompts/$EVAL_PROMPT_DIR"
if [ ! -d "$EVAL_PROMPT_PATH" ]; then
    echo "Error: Evaluation prompt directory $EVAL_PROMPT_PATH does not exist"
    echo "Available evaluation prompt directories:"
    ls -1 motivated_reasoning/evaluation/prompts/*/ 2>/dev/null | sed 's/.*\///' | sed 's/\/$//' || echo "  No evaluation prompt directories found"
    exit 1
fi

# Check if prompt.txt and suffix.txt exist
if [ ! -f "$EVAL_PROMPT_PATH/prompt.txt" ]; then
    echo "Error: prompt.txt not found in $EVAL_PROMPT_PATH"
    exit 1
fi

if [ ! -f "$EVAL_PROMPT_PATH/suffix.txt" ]; then
    echo "Error: suffix.txt not found in $EVAL_PROMPT_PATH"
    exit 1
fi

# Find all iteration directories (folders that match iteration-* pattern)
ITERATION_DIRS=($(find "$INFERENCE_PROMPT_PATH" -maxdepth 1 -type d -name "iteration-*" | sort -V))
if [ ${#ITERATION_DIRS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $INFERENCE_PROMPT_PATH"
    exit 1
fi

# Extract iteration identifiers from directory names
ITERATIONS=()
for dir in "${ITERATION_DIRS[@]}"; do
    dirname=$(basename "$dir")
    # Extract iteration identifier from directory name like "iteration-7" or "iteration-base"
    iteration=$(echo "$dirname" | sed -n 's/iteration-\(.*\)/\1/p')
    if [ ! -z "$iteration" ]; then
        ITERATIONS+=("$iteration")
    fi
done

# Remove duplicates and sort (base will sort first alphabetically, then numbers)
ITERATIONS=($(printf "%s\n" "${ITERATIONS[@]}" | sort -u))

if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No valid iterations found in $INFERENCE_PROMPT_PATH"
    exit 1
fi

echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[@]}"

# Run all iterations
echo "Running all iterations: ${ITERATIONS[@]}"

echo "Run name: $RUN_NAME"
echo "Inference prompt directory: $INFERENCE_PROMPT_DIR"
echo "Evaluation prompt directory: $EVAL_PROMPT_DIR"
echo "Evaluation target: $EVAL_TARGET"
echo "Evaluator model: $EVALUATOR_MODEL"
if [ ${#FILTERED_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${FILTERED_ARGS[@]}"
fi
echo ""

# SLURM configuration - Need GPU for local model inference
SLURM_CONFIG="--cpus-per-task=4 --mem=32G --gres=gpu:1 --time=2:00:00"

echo "Submitting SLURM jobs for local model-based inference output evaluation on iterations: ${ITERATIONS[@]}"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting local model evaluation job for iteration $iteration..."

    # Create job name - sanitize the model name for job naming
    MODEL_SHORT=$(echo "$EVALUATOR_MODEL" | sed 's/\//-/g' | sed 's/meta-llama-//g' | sed 's/Meta-Llama-//g' | cut -c1-20)
    job_name="local_eval_${RUN_NAME}_iter${iteration}_${INFERENCE_PROMPT_DIR}_${MODEL_SHORT}"
    extra_args_string=""
    if [ ${#FILTERED_ARGS[@]} -gt 0 ]; then
        extra_args_string="$(printf ' %q' "${FILTERED_ARGS[@]}")"
    fi

    # Submit SLURM job
    sbatch $SLURM_CONFIG \
        --job-name="$job_name" \
        --output="slurm_logging/${job_name}_%j.out" \
        --error="slurm_logging/${job_name}_%j.err" \
        << EOF
#!/bin/bash
#SBATCH --job-name="$job_name"

# Source bash config, which also does conda
source /nas/ucb/nikihowe/config/bashrc
conda activate motivated_reasoning_env

# Change to project directory
cd /nas/ucb/nikihowe/motivated-reasoning

# Run the local model evaluation script
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    --iteration $iteration \
    --inference_prompt_dir $INFERENCE_PROMPT_DIR \
    --eval_prompt_dir $EVAL_PROMPT_DIR \
    --eval_target $EVAL_TARGET \
    --evaluator_model $EVALUATOR_MODEL$extra_args_string

echo "Completed local model-based inference output evaluation for iteration $iteration with inference prompt $INFERENCE_PROMPT_DIR and evaluation prompt $EVAL_PROMPT_DIR"
EOF

    echo "Submitted local model evaluation job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All local model evaluation SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Evaluation results will be saved in: evaluation_output/$RUN_NAME/$INFERENCE_PROMPT_DIR/$EVAL_TARGET/evaluator-[model-name]/$EVAL_PROMPT_DIR/iteration-X/"
