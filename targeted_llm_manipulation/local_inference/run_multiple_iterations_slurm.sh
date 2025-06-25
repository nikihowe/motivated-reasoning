#!/bin/bash

# Script to submit SLURM jobs for multiple iterations of HarmBench inference
# Usage: ./run_multiple_iterations_slurm.sh [custom_run_name]

# Default run name
DEFAULT_RUN_NAME="harmbench_kto_long_lr_5e-5-06_20_113158"

# Use provided run name or default
RUN_NAME="${1:-$DEFAULT_RUN_NAME}"

MODEL_PATH="/nas/ucb/nikihowe/chai_motivated_reasoning/data/models"
SCRIPT_PATH="targeted_llm_manipulation/local_inference/run_harmbench_inference.py"

# Automatically detect iterations by scanning the model directory
MODEL_DIR="$MODEL_PATH/$RUN_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist"
    exit 1
fi

# Find all iteration directories (folders that are numbers)
ITERATIONS=($(find "$MODEL_DIR" -maxdepth 1 -type d -name "[0-9]*" | sort -n | xargs -n1 basename))
if [ ${#ITERATIONS[@]} -eq 0 ]; then
    echo "Error: No iteration directories found in $MODEL_DIR"
    exit 1
fi

echo "Found ${#ITERATIONS[@]} iterations: ${ITERATIONS[@]}"

# Define node names (without domain)
# We omit "vae" and "ppo" because they are A4000s which don't have enough memory
NODE_NAMES=("dqn" "ddpg" "gail" "gan" "airl" "sac" "cirl" "rlhf")

# Construct full nodelist with domain
NODELIST=$(IFS=, ; echo "${NODE_NAMES[*]/%/.ist.berkeley.edu}")

# Debug: show the constructed nodelist
echo "Constructed nodelist: '$NODELIST'"

# SLURM configuration
SLURM_CONFIG="--partition=main --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=0:20:00 --nodelist=$NODELIST"

echo "Submitting SLURM jobs for iterations: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Model path: $MODEL_PATH"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting job for iteration $iteration..."
    
    # Create job name
    job_name="harmbench_${RUN_NAME}_iter${iteration}"
    
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
cd /nas/ucb/nikihowe/chai_motivated_reasoning

# Run the inference script
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    --iteration $iteration \
    --model_path $MODEL_PATH

echo "Completed inference for iteration $iteration"
EOF

    echo "Submitted job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/" 