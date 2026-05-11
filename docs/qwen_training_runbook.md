# Qwen3 Risky-COT Training Runbook

This is the working record for running `Qwen/Qwen3-8B` through the risky-cot KTO
training setup. It documents the code assumptions, launch commands, resume behavior,
and checks that matter for reproducing or extending the run.

## Goal

Run the same risky-cot experiment semantics as the paper-range Llama runs, but with
`Qwen/Qwen3-8B` as both agent and environment model.

The Qwen config intentionally changes model/backend formatting details, not the
experiment prompt logic:

- Config: `motivated_reasoning/config/experiment_configs/risky-cot/risky-cot-25-qwen.yaml`
- Parent config: `motivated_reasoning/config/experiment_configs/risky-cot/base/base_risky.yaml`
- Run name prefix: `risky_qwen`
- RL iterations: `10`
- Reasoning tag: `<think>...</think>`
- Assistant completion format: Qwen chat format
- Ground-truth scoring: `true`
- Veto/influence prompt type: `five_point`

## Qwen-Specific Code Paths

The Qwen support depends on these code paths:

- `motivated_reasoning/reasoning_tags.py`
  - Renders configured reasoning tags.
  - Parses both `<thinking>` and `<think>` for backwards compatibility.
- `motivated_reasoning/config/experiment_config.py`
  - Adds `reasoning_tag` and `assistant_completion_format` to experiment/training args.
- `motivated_reasoning/trajectory_generator/trajectory_generator.py`
  - Renders the agent system prompt with the configured reasoning tag.
- `motivated_reasoning/environment/assessor_model.py`
  - Applies formatting penalties using the configured reasoning tag, while accepting aliases.
- `motivated_reasoning/inference/model_backends.py`
  - HF generation slices output by prompt length, rather than model-specific assistant markers.
- `motivated_reasoning/RL/run_KTO_iteration.py`
  - Uses `processing_class=tokenizer` for current TRL.
  - Uses manual Qwen assistant completion formatting so `<think>...</think>` is preserved.

## Validation Before Launch

Useful smoke checks:

```bash
/nas/ucb/nikihowe/conda/envs/motivated_reasoning_env/bin/python -m pytest tests/test_reasoning_tags.py

/nas/ucb/nikihowe/conda/envs/motivated_reasoning_env/bin/python \
  motivated_reasoning/training/launch_training.py \
  --config risky-cot-25-qwen.yaml \
  --gpus 0 \
  --only-load-config
```

Expected config-load checks:

- `model_names.agent == Qwen/Qwen3-8B`
- `model_names.env == Qwen/Qwen3-8B`
- `reasoning_tag == "think"`
- `assistant_completion_format == "qwen"`

## Launch

Use the SLURM autocopy wrapper so the submitted job runs against a stable copy of
the code:

```bash
CONDA_DEFAULT_ENV=motivated_reasoning_env \
PATH=/nas/ucb/nikihowe/conda/envs/motivated_reasoning_env/bin:$PATH \
bash motivated_reasoning/training/slurm/autocopy_and_sbatch.sh \
  --config-name risky-cot-25-qwen \
  --cpus 8 \
  --mem 140gb \
  --gpus 8 \
  --gpu-type noshards \
  --time 9:00:00 \
  --qos default
```

The launcher prints a pre-submit config check. For an 8-GPU job, it should show:

- `Using all available CUDA devices [0, 1, 2, 3, 4, 5, 6, 7]`
- `Accelerate training on GPUs: [0, 1, 2, 3, 4, 5, 6, 7]`
- `Set gradient_accumulation_steps to 2`

On 4 GPUs, gradient accumulation is `4`. This preserves effective batch size `32`.

## Resume

`autocopy_and_sbatch.sh` supports an optional `--timestamp` argument. Use it to
resume the same run instead of creating a new timestamp:

```bash
CONDA_DEFAULT_ENV=motivated_reasoning_env \
PATH=/nas/ucb/nikihowe/conda/envs/motivated_reasoning_env/bin:$PATH \
bash motivated_reasoning/training/slurm/autocopy_and_sbatch.sh \
  --config-name risky-cot-25-qwen \
  --cpus 8 \
  --mem 100gb \
  --gpus 4 \
  --gpu-type noshards \
  --time 15:00:00 \
  --qos default \
  --timestamp 05_10_160840
```

Resume behavior lives in `motivated_reasoning/RL/base_iteration.py`.

Important details:

- Completed checkpoints are under `data/models/<run>/<iteration>/checkpoint-*`.
- Completed trajectories are under `data/trajectories/<run>/<iteration>/`.
- A completed trajectory iteration has `selected_trajectories.jsonl`.
- If a partial next trajectory directory exists, resume deletes it and regenerates that iteration.
- If a job dies between trajectory selection and checkpoint creation, inspect the partially completed
  iteration before relaunching.

For the active May 10 run:

- Run: `risky_qwen-05_10_160840`
- Completed checkpoint 1: `data/models/risky_qwen-05_10_160840/1/checkpoint-7`
- Initial 4-GPU job: `1132676`, cancelled during iteration 2 generation.
- 8-GPU resume job: `1132711`, cancelled while pending because SLURM estimated a multi-day wait.
- 4-GPU, 15-hour resume job: `1132730`.

## Monitoring

Queue:

```bash
squeue -j <job_id> -o '%.18i %.9P %.40j %.8u %.2t %.10M %.10l %.12q %.24R %.30b'
```

Logs:

```bash
tail -f slurm_logging/risky-cot-25-qwen_05_10_160840-<job_id>.out
```

Checkpoint directories:

```bash
find data/models/risky_qwen-05_10_160840 -maxdepth 2 -type d | sort
find data/trajectories/risky_qwen-05_10_160840 -maxdepth 2 -type d | sort
```

Expected iteration structure:

1. Generate `1600` trajectories.
2. Select top/bottom `100` trajectories each.
3. Save `200` KTO examples to `trajectories_for_train.jsonl`.
4. Run KTO for `7` optimizer steps.
5. Save `checkpoint-7`.

## Speed Notes

The expensive step is rollout generation, not KTO.

Observed 4-GPU Qwen timings:

- Iteration 0 generation: `54:05`
- Iteration 1 generation: `1:13:58`
- KTO training: about `112-136s`

With `separate_agent_env_devices: "no"`, trajectory generation starts one worker per visible GPU.
The total trajectory count stays fixed at `1600`, so 8 GPUs should roughly halve generation time,
modulo stragglers, CPU overhead, and shared filesystem load.

## Output Inspection

Qwen iteration-0 outputs were much longer than the comparable Llama outputs.

Observed comparison:

- Qwen usually emits `<think>...</think><answer>...</answer>` correctly when it finishes.
- Qwen sometimes spends too much of the `1024` completion-token budget inside `<think>`.
- Llama is shorter but often omits `</answer>`.

Before changing prompts or token budgets, first inspect whether later iterations reduce the rate of
unparsed/cut-off Qwen answers.

## Related Docs

- `EXPERIMENT_LOG.md`: chronological record of launches and job IDs.
- `CONTEXT.md`: high-level repo and paper-reproduction map.
- `scripts/reproduce_paper_plots.sh`: regenerate paper plots from existing data.
- `scripts/reproduce_paper_eval_data.sh`: dry-run or launch the eval/analysis commands needed by paper plots.
