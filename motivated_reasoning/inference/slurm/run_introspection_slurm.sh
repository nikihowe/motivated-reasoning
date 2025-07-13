#!/bin/bash

# Script to submit SLURM jobs for multiple iterations of introspection (safety and reasoning)
# Usage: ./run_introspection_slurm.sh [custom_run_name]

# Default run name
DEFAULT_RUN_NAME="harmbench_kto_motivated-06_25_163944"

# Use provided run name or default
RUN_NAME="${1:-$DEFAULT_RUN_NAME}"

# Accept extra flags for python script
EXTRA_FLAGS="${@:2}"

MODEL_PATH="/nas/ucb/nikihowe/chai_motivated_reasoning/data/models"
SCRIPT_PATH="motivated_reasoning/inference/run_local/run_introspection.py"

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

# SLURM configuration - lighter resources since self-evaluation is simpler
SLURM_CONFIG="--partition=main --gpus=A6000:1 --cpus-per-task=4 --mem=16G --time=0:10:00"

echo "Submitting SLURM jobs for introspection on iterations: ${ITERATIONS[@]}"
echo "Model: $RUN_NAME"
echo "Model path: $MODEL_PATH"
echo ""

for iteration in "${ITERATIONS[@]}"; do
    echo "Submitting introspection job for iteration $iteration..."
    
    # Create job name
    job_name="introspection_${RUN_NAME}_iter${iteration}"
    
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

# Run the introspection script
python $SCRIPT_PATH \
    --run_name $RUN_NAME \
    --iteration $iteration \
    --model_path $MODEL_PATH \
    $EXTRA_FLAGS

echo "Completed introspection for iteration $iteration"
EOF

    echo "Submitted introspection job for iteration $iteration with job name: $job_name"
    echo ""
done

echo "All introspection SLURM jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: slurm_logging/"
echo "Introspection results will be saved in: introspection_output/$RUN_NAME/" 