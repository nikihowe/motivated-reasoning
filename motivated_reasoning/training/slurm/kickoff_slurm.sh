#!/bin/bash

###############################################################
# PARAMETERS

# Default config name if none provided
DEFAULT_CONFIG_NAMES="harmbench_static_harmful"

# Check if config names were provided as command line argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_names>"
    echo "Example: $0 'harmbench_static_harmful'"
    echo "Example: $0 'harmbench_static_harmful therapy'"
    echo "Using default config: $DEFAULT_CONFIG_NAMES"
    CONFIG_NAMES="$DEFAULT_CONFIG_NAMES"
else
    CONFIG_NAMES="$1"
fi

# SLURM job parameters
SLURM_CPUS_PER_TASK=4  # down from 8
SLURM_MEM="70gb" # May require up to 100gb per GPU for bigger models  # down from 100
SLURM_GPUS="4" # IF USING MORE GPUS, NEED TO INCREASE MEMORY PROPORTIONALLY
GPU_TYPE="noshards" # A100 (faster generation) or A6000 (often more available), "either" (for either A100 or A6000), 
#or "all" (for all available GPUs, will break most jobs). PCI for A100-PCI and SXM4 for A100-SXM4. SXM4 should be used for training LLama-3.1-70B
SLURM_TIME="9:00:00"
SLURM_QOS="high" # can set to high if this is blocking your progress and you only need one/two jobs to run

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Loop through each config name and run autocopy_and_sbatch.sh for each one
for CONFIG_NAME in $CONFIG_NAMES; do
    bash $SCRIPT_DIR/autocopy_and_sbatch.sh --config-name "$CONFIG_NAME" --cpus "$SLURM_CPUS_PER_TASK" --mem "$SLURM_MEM" --gpus "$SLURM_GPUS" --gpu-type "$GPU_TYPE" --time "$SLURM_TIME" --qos "$SLURM_QOS"
done