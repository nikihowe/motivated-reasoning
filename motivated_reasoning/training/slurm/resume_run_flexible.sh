#!/bin/bash

###############################################################
# USAGE: ./resume_run_flexible.sh <model_name>
# Example: ./resume_run_flexible.sh safe-09_17_213317
###############################################################

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a model name"
    echo "Usage: $0 <model_name>"
    echo "Example: $0 safe-09_17_213317"
    exit 1
fi

# Extract model name and timestamp from the argument
MODEL_NAME=$1
echo "Resuming model: $MODEL_NAME"

# Extract the timestamp from the model name (assumes format: prefix-MM_DD_HHMMSS)
if [[ $MODEL_NAME =~ -([0-9]{2}_[0-9]{2}_[0-9]{6})$ ]]; then
    TIMESTAMP="${BASH_REMATCH[1]}"
    CONFIG_PREFIX="${MODEL_NAME%-*}"  # Everything before the last dash
else
    echo "Error: Could not extract timestamp from model name '$MODEL_NAME'"
    echo "Expected format: prefix-MM_DD_HHMMSS (e.g., safe-09_17_213317)"
    exit 1
fi

echo "Extracted config prefix: $CONFIG_PREFIX"
echo "Extracted timestamp: $TIMESTAMP"

# Map model prefixes to config names
case $CONFIG_PREFIX in
    "safe")
        CONFIG_NAME="safe-cot"
        ;;
    "risky")
        CONFIG_NAME="risky-cot"
        ;;
    "now")
        CONFIG_NAME="now-cot"
        ;;
    "later")
        CONFIG_NAME="later-cot"
        ;;
    "fewer-apples")
        CONFIG_NAME="fewer-apples-cot"
        ;;
    "more-apples")
        CONFIG_NAME="more-apples-cot"
        ;;
    "therapy")
        CONFIG_NAME="therapy"
        ;;
    *)
        echo "Error: Unknown model prefix '$CONFIG_PREFIX'"
        echo "Supported prefixes: safe, risky, now, later, fewer-apples, more-apples, therapy"
        exit 1
        ;;
esac

echo "Using config: $CONFIG_NAME"

# SLURM job parameters (matching kickoff_slurm.sh)
export SLURM_CPUS_PER_TASK=8
export SLURM_MEM="100gb" # May require up to 100gb per GPU for bigger models
export SLURM_GPUS="4" # IF USING MORE GPUS, NEED TO INCREASE MEMORY PROPORTIONALLY
export GPU_TYPE="either" # A100, A6000, or "either"
export SLURM_TIME="6:00:00"
export SLURM_QOS="default"

###############################################################

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FILE_TO_RUN="launch_training.py"

# Check if /nas/ directory exists to determine if we're on the CHAI cluster
if [ -d "/nas" ]; then
    PROJ_DIR="/nas/ucb/$(whoami)/motivated-reasoning"

    if [ "$GPU_TYPE" == "A100" ]; then
        NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu"
    elif [ "$GPU_TYPE" == "A6000" ]; then
        NODE_LIST="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
    elif [ "$GPU_TYPE" == "either" ]; then
        NODE_LIST="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu,ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu,gail.ist.berkeley.edu,gan.ist.berkeley.edu"
    else
        echo "Invalid GPU type: $GPU_TYPE"
        exit 1
    fi

    NODE_PARAM="--nodelist=$NODE_LIST"
    MEM_PARAM="#SBATCH --mem=$SLURM_MEM"
    QOS="#SBATCH --qos=$SLURM_QOS"
    if [ "$SLURM_QOS" == "scavenger" ]; then
        QOS="$QOS --partition scavenger"
    fi
else
    PROJ_DIR="$HOME/motivated-reasoning"
    NODE_PARAM="--partition=single"
    MEM_PARAM=""
    QOS=""
fi

# Verify the model exists
MODEL_DIR="$PROJ_DIR/data/models/$MODEL_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo "Available models:"
    ls -1 "$PROJ_DIR/data/models/" | grep -E "^[a-zA-Z-]+-[0-9]{2}_[0-9]{2}_[0-9]{6}$" | head -10
    exit 1
fi

echo "Found model directory: $MODEL_DIR"

# Generate job name and temp directory
JOB_NAME="${MODEL_NAME}_resume"
TEMP_DIR="$PROJ_DIR/tmp/tmp_${MODEL_NAME}_resume"

# Fixed SLURM params
export SLURM_NODES=1
export SLURM_NTASKS_PER_NODE=1
export SLURM_OUTPUT="$PROJ_DIR/slurm_logging/$JOB_NAME-%j.out"

# Check for correct Conda environment
if [[ "$CONDA_DEFAULT_ENV" != "motivated_reasoning_env" ]]; then
    echo "Error: Not in the 'motivated_reasoning_env' Conda environment. Please activate it before running this script."
    exit 1
fi

echo "Using Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Define the original project directory
ORIGINAL_DIR="$PROJ_DIR/motivated_reasoning"

# Create a unique temporary directory and copy the project to it
echo "Creating temporary directory: $TEMP_DIR"
mkdir -p $TEMP_DIR
cp -r $ORIGINAL_DIR $TEMP_DIR

# Modify the import statements in the tmp copy
cd $TEMP_DIR/motivated_reasoning
python utils/prep_for_slurm.py . $FILE_TO_RUN

cat << EOF > $JOB_NAME
#!/bin/bash
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --cpus-per-task=$SLURM_CPUS_PER_TASK
#SBATCH --gpus=$SLURM_GPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks-per-node=$SLURM_NTASKS_PER_NODE
#SBATCH $NODE_PARAM
$MEM_PARAM
$QOS

export NCCL_P2P_LEVEL=NVL
conda activate motivated_reasoning_env
echo "Conda environment: \$CONDA_DEFAULT_ENV"

# Get the file to run and the temporary directory from command-line arguments
FILE_TO_RUN=\$1
TEMP_DIR=\$2/motivated_reasoning
CONFIG_NAME=\$3
TIMESTAMP=\$4

# Change to the temporary directory
cd \$TEMP_DIR

# Run the Python script with resume timestamp
python training/\$FILE_TO_RUN --config \$CONFIG_NAME.yaml --all-gpus --timestamp \$TIMESTAMP

# Optional: Clean up the temporary directory after the job finishes
# Uncomment the following line if you want to automatically delete the temporary directory
# rm -rf \$TEMP_DIR
EOF

# Run the SLURM job
echo "Command to run: python training/$FILE_TO_RUN --config $CONFIG_NAME.yaml --all-gpus --timestamp $TIMESTAMP"
echo "Submitting SLURM job..."
sbatch $JOB_NAME $FILE_TO_RUN $TEMP_DIR $CONFIG_NAME $TIMESTAMP
