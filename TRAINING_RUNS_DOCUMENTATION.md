# Training Runs Documentation

## Overview

This document describes the training infrastructure for the motivated reasoning experiments, specifically focusing on the v2 datasets and the configuration system for launching training runs with different models (Llama and Qwen) and learning rates.

## Directory Structure

```
motivated_reasoning/config/experiment_configs/
├── later-cot_v2/
│   ├── base/
│   │   └── base_later_v2.yaml          # Base config for later-cot v2
│   ├── later-cot_v2.yaml               # Llama, 2.5e-5 LR, 20 iter
│   ├── later-cot_v2-qwen.yaml          # Qwen, 2.5e-5 LR, 20 iter
│   ├── later-cot_v2-15e5.yaml          # Llama, 1.5e-5 LR, 20 iter
│   └── later-cot_v2-qwen-15e5.yaml     # Qwen, 1.5e-5 LR, 20 iter
├── now-cot_v2/
│   ├── base/
│   │   └── base_now_v2.yaml
│   ├── now-cot_v2.yaml
│   ├── now-cot_v2-qwen.yaml
│   ├── now-cot_v2-15e5.yaml
│   └── now-cot_v2-qwen-15e5.yaml
├── risky-cot_v2/
│   ├── base/
│   │   └── base_risky_v2.yaml
│   ├── risky-cot_v2.yaml
│   ├── risky-cot_v2-qwen.yaml
│   ├── risky-cot_v2-15e5.yaml
│   └── risky-cot_v2-qwen-15e5.yaml
├── safe-cot_v2/
│   ├── base/
│   │   └── base_safe_v2.yaml
│   ├── safe-cot_v2.yaml
│   ├── safe-cot_v2-qwen.yaml
│   ├── safe-cot_v2-15e5.yaml
│   └── safe-cot_v2-qwen-15e5.yaml
└── harmbench-cot-tags/
    ├── base/
    │   └── base_tags.yaml
    ├── harmbench_cot_tags.yaml
    ├── harmbench_cot_tags-qwen.yaml          # 2.5e-5 LR, 20 iter
    └── harmbench_cot_tags-qwen-15e5.yaml     # 1.5e-5 LR, 20 iter
```

## Configuration System

### Base Configs

Each experiment type has a base configuration file that contains all the default parameters:
- `base_later_v2.yaml`
- `base_now_v2.yaml`
- `base_risky_v2.yaml`
- `base_safe_v2.yaml`
- `base_tags.yaml` (for harmbench)

Base configs specify:
- Model names (default: `meta-llama/Meta-Llama-3-8B-Instruct`)
- Environment class and dataset names
- Training hyperparameters (batch size, gradient checkpointing, etc.)
- Default learning rate and iterations
- KTO-specific settings (beta, target_ratio, etc.)

### Override Configs

Each specific training run has its own config file that overrides the base config with specific parameters:

**Key parameters that are typically overridden:**
- `run_name`: Unique name for the training run
- `iterations`: Number of training iterations (we use 20)
- `learning_rate`: Learning rate for training (2.5e-5 or 1.5e-5)
- `model_names`: For Qwen runs, overrides to use `Qwen/Qwen3-8B`
- `veto_prompt_type`: Set to `"five_point"`
- `formatting_penalty_scale_factor`: Set to `0.1` for v2 datasets, `1` for harmbench
- `use_ground_truth_scoring`: Set to `true` for v2 datasets, `false` for harmbench

### Example Config Structure

```yaml
parent_config_to_override: "base_later_v2.yaml"
run_name: "later_v2_qwen_15e5"

# Trajectory generation settings
iterations: 20

# Training settings
learning_rate: 1.5e-5

# Influence model settings
veto_prompt_type: "five_point"

# Formatting penalty settings
formatting_penalty_scale_factor: 0.1

# Ground truth settings
use_ground_truth_scoring: true

# Qwen-specific settings (only for Qwen configs)
model_names:
  agent: "Qwen/Qwen3-8B"
  env: "Qwen/Qwen3-8B"

env_class: "later-cot_v2"
```

## Datasets

### V2 Datasets
The v2 datasets have ground truth scoring enabled and use the following environments:
- `later_train_v2` - Constitutional AI evaluator sees agent reasoning before making decision
- `now_train_v2` - Constitutional AI evaluator sees agent reasoning during decision
- `risky_train_v2` - Risky/harmful scenario training
- `safe_train_v2` - Safe scenario training

### Harmbench Dataset
- `harmbench-static-train-cot` - Static harmbench dataset with chain-of-thought
- Does NOT use ground truth scoring
- Different formatting penalty (1.0 vs 0.1 for v2 datasets)

## Launching Training Runs

### Basic Usage

```bash
# Launch a single training run with high priority
./motivated_reasoning/training/slurm/kickoff_slurm.sh "config_name"

# Launch multiple training runs with high priority
./motivated_reasoning/training/slurm/kickoff_slurm.sh "config1 config2 config3"

# Launch with normal/default priority
SLURM_QOS=default ./motivated_reasoning/training/slurm/kickoff_slurm.sh "config1 config2"
```

### Config Naming Convention

The config name is the filename without the `.yaml` extension:
- `later-cot_v2` → uses `later-cot_v2.yaml`
- `later-cot_v2-qwen` → uses `later-cot_v2-qwen.yaml`
- `later-cot_v2-15e5` → uses `later-cot_v2-15e5.yaml`

### SLURM Parameters

Default parameters (defined in `kickoff_slurm.sh`):
- CPUs per task: 4
- Memory: 70GB
- GPUs: 4
- GPU type: noshards
- Time limit: 20:00:00
- QOS: high (can be overridden with `SLURM_QOS` env var)

## Recent Training Runs (January 26, 2026)

### Batch 1: 2.5e-5 Learning Rate (High Priority)

**Qwen v2 runs:**
- Job 1024507: `later-cot_v2-qwen` (Qwen, 2.5e-5 LR, 20 iter)
- Job 1024508: `now-cot_v2-qwen` (Qwen, 2.5e-5 LR, 20 iter)
- Job 1024509: `risky-cot_v2-qwen` (Qwen, 2.5e-5 LR, 20 iter)
- Job 1024510: `safe-cot_v2-qwen` (Qwen, 2.5e-5 LR, 20 iter)

**Llama v2 runs:**
- Job 1024511: `later-cot_v2` (Llama, 2.5e-5 LR, 20 iter)
- Job 1024512: `now-cot_v2` (Llama, 2.5e-5 LR, 20 iter)
- Job 1024513: `risky-cot_v2` (Llama, 2.5e-5 LR, 20 iter)
- Job 1024514: `safe-cot_v2` (Llama, 2.5e-5 LR, 20 iter)

**Harmbench run:**
- Job 1024515: `harmbench_cot_tags-qwen` (Qwen, 2.5e-5 LR, 20 iter)

### Batch 2: 1.5e-5 Learning Rate (Default Priority)

**Llama v2 runs:**
- Job 1024516: `later-cot_v2-15e5` (Llama, 1.5e-5 LR, 20 iter)
- Job 1024517: `now-cot_v2-15e5` (Llama, 1.5e-5 LR, 20 iter)
- Job 1024518: `risky-cot_v2-15e5` (Llama, 1.5e-5 LR, 20 iter)
- Job 1024519: `safe-cot_v2-15e5` (Llama, 1.5e-5 LR, 20 iter)

**Qwen v2 runs:**
- Job 1024520: `later-cot_v2-qwen-15e5` (Qwen, 1.5e-5 LR, 20 iter)
- Job 1024521: `now-cot_v2-qwen-15e5` (Qwen, 1.5e-5 LR, 20 iter)
- Job 1024522: `risky-cot_v2-qwen-15e5` (Qwen, 1.5e-5 LR, 20 iter)
- Job 1024523: `safe-cot_v2-qwen-15e5` (Qwen, 1.5e-5 LR, 20 iter)

**Harmbench run:**
- Job 1024524: `harmbench_cot_tags-qwen-15e5` (Qwen, 1.5e-5 LR, 20 iter)

**Total: 18 training runs**
- 8 Qwen runs (4 v2 + harmbench at 2.5e-5, 4 v2 + harmbench at 1.5e-5)
- 8 Llama runs (4 v2 at 2.5e-5, 4 v2 at 1.5e-5)
- 2 harmbench runs (1 at 2.5e-5, 1 at 1.5e-5)

## Monitoring Training Jobs

```bash
# Check all your jobs
squeue -u $USER

# Check specific job status
squeue -j <job_id>

# Check only the recent training jobs
squeue -u $USER | grep -E "1024(50[7-9]|51[0-9]|52[0-4])"

# View job details
scontrol show job <job_id>

# Check job output logs (after job starts)
# Logs are typically in the tmp directory that was created
ls -lt tmp/
```

## Training Infrastructure Components

### Main Scripts
- `motivated_reasoning/training/slurm/kickoff_slurm.sh` - Main entry point for launching training
- `motivated_reasoning/training/slurm/autocopy_and_sbatch.sh` - Creates temp directory and submits job
- `motivated_reasoning/training/launch_training.py` - Python script that runs the actual training

### Key Features
- **Config inheritance**: Override configs inherit from base configs
- **Automatic temp directory creation**: Each run gets a unique temp directory
- **DeepSpeed integration**: Uses DeepSpeed for distributed training
- **LoRA training**: All models use LoRA (r=16, alpha=32, dropout=0.1)
- **KTO training**: Uses KTO (Kahneman-Tversky Optimization) algorithm
- **Gradient checkpointing**: Enabled to reduce memory usage
- **WandB logging**: Logs to Weights & Biases for experiment tracking

## Key Differences Between Configurations

### V2 vs Non-V2 Datasets
- V2 datasets have `use_ground_truth_scoring: true`
- V2 datasets use different environment datasets (`*_train_v2`)
- V2 datasets typically use `formatting_penalty_scale_factor: 0.1`

### Llama vs Qwen
- Llama uses `meta-llama/Meta-Llama-3-8B-Instruct`
- Qwen uses `Qwen/Qwen3-8B`
- Both use the same training infrastructure and hyperparameters

### Harmbench vs V2
- Harmbench uses `formatting_penalty_scale_factor: 1` (not 0.1)
- Harmbench uses `use_ground_truth_scoring: false`
- Harmbench uses different dataset (`harmbench-static-train-cot`)
- Harmbench has `allow_id_to_see_cot: false` (v2 has `true`)

## Common Training Parameters

All current training runs use:
- **Iterations**: 20
- **Learning rate**: 2.5e-5 or 1.5e-5
- **Veto prompt type**: five_point
- **Batch size**: 2 per device, effective batch size 32
- **Gradient accumulation steps**: 2 (auto-calculated)
- **Max length**: 4096
- **LoRA rank**: 16
- **KTO beta**: 0.1
- **KTO target ratio**: 1.05
- **Learning rate decay**: 0.9 multiplier across iterations
- **Optimizer**: AdamW (torch)
- **LR scheduler**: constant

## Creating New Configs

To create a new training configuration:

1. **Choose the base config** that matches your experiment type
2. **Create a new YAML file** in the appropriate directory
3. **Set the parent config**: `parent_config_to_override: "base_*.yaml"`
4. **Override the necessary parameters**:
   - `run_name`: Unique identifier
   - `iterations`: Number of training iterations
   - `learning_rate`: Learning rate
   - `model_names`: For non-Llama models
   - Any other parameters you want to change

Example:
```yaml
parent_config_to_override: "base_later_v2.yaml"
run_name: "my_experiment"
iterations: 20
learning_rate: 3e-5
# ... other overrides
```

## Troubleshooting

### Job stays in pending (PD) state
- Check if you've hit GPU limits: `squeue -u $USER | grep " R "` (count running jobs)
- High priority jobs may be blocked by QOS limits
- Try using default priority: `SLURM_QOS=default ./motivated_reasoning/training/slurm/kickoff_slurm.sh`

### Job fails immediately
- Check the SLURM output log in the temp directory
- Verify the config file exists and is valid YAML
- Ensure the parent config reference is correct

### Out of memory errors
- Reduce `per_device_train_batch_size` in base config
- Ensure `gradient_checkpointing: true` is enabled
- May need to increase `SLURM_MEM` in kickoff script

### Config not found
- Verify the config name matches the filename (without .yaml)
- Check that the file is in the correct directory
- Make sure parent config path is relative to the config file location

## Future Enhancements

Potential improvements to consider:
- Add support for different iteration counts via command line
- Create a config generation script for systematic sweeps
- Add resume functionality for interrupted training runs
- Implement automatic hyperparameter search
- Add validation metrics tracking during training
