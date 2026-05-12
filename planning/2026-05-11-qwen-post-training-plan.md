# Qwen Post-Training Plan -- 2026-05-11

This is the next-step checklist for when the currently queued/running Qwen training jobs finish.

## Active Training Runs

Full 10-iteration Qwen setting runs:

| Setting | Run prefix | Timestamp | SLURM job |
|---------|------------|-----------|-----------|
| safe-cot | `safe_qwen` | `05_11_203401` | `1132943` |
| now-cot | `now_qwen` | `05_11_203402` | `1132944` |
| later-cot | `later_qwen` | `05_11_203403` | `1132945` |
| harmbench-cot-tags | `harmbench_cot_tags_qwen` | `05_11_203404` | `1132947` |

Five-iteration risky Qwen visible-reasoning ablations:

| Ablation | Run prefix | Timestamp | SLURM job |
|----------|------------|-----------|-----------|
| Dataset-only | `risky_qwen_dataset` | `05_11_214101` | `1133027` |
| System-prompt-only | `risky_qwen_sp` | `05_11_214102` | `1133028` |
| All fixes | `risky_qwen_allfixes` | `05_11_214104` | `1133029` |
| Min-reasoning-only | `risky_qwen_minthink` | `05_11_214103` | `1133030` |

## First Checks After Training

1. Confirm each run reached its intended final iteration.
2. Check SLURM logs for crashes, OOMs, checkpoint save failures, or W&B/logging problems.
3. Record final run names, checkpoint paths, W&B links, and any failures in `EXPERIMENT_LOG.md`.

## Visible-Reasoning Audit

Run this before spending evaluation compute.

For each Qwen run, especially the four ablations, measure:

- percentage of selected trajectories with empty visible `<think></think>`
- malformed tag rate
- median and mean reasoning word count
- answer reward by iteration
- a few raw trajectory samples from early and final iterations

Compare against the original risky pilot:

- `risky_qwen-05_10_160840`

Main decision question: did dataset cleanup, system-prompt clarification, min-reasoning penalty, or
all fixes preserve nonempty visible reasoning while keeping answer-reward learning?

## Decide What To Fully Evaluate

Prioritize full inference/evaluation only for runs that both:

- learned the answer objective, and
- preserve enough nonempty visible reasoning to make CoT/monitor analyses meaningful.

If a run still has mostly empty reasoning, keep it as a negative result but do not automatically spend
full evaluation compute on it.

## Paper-Style Inference

For useful completed risky-style Qwen runs, run paper-style inference against the opposing/target
constitution prompt. For risky-trained Qwen runs, the likely inference prompt is:

- `safe_constitutional_cot_v2_qwen`

Use Qwen-specific settings:

- `reasoning_tag: think`
- `assistant_completion_format: qwen`
- enough max tokens that visible reasoning is not truncated

For full setting runs, mirror the paper mappings:

- `safe_qwen-*` -> `risky_constitutional_cot_v2_qwen`
- `now_qwen-*` -> `later_constitutional_cot_v2_qwen`
- `later_qwen-*` -> `now_constitutional_cot_v2_qwen`
- `harmbench_cot_tags_qwen-*` -> `harmbench_constitutional_cot_v2_qwen`

## Evaluations

For inference outputs, run:

- correctness eval: `simple_correct_choice`
- motivated-reasoning / constitution-use eval: `copy_constitution_motivated_reasoning_v3`
- rebuttal robustness eval if needed: `copy_constitution_motivated_reasoning_v3_for_rebuttal`
- local monitor eval, response-only target
- local monitor eval, reasoning+response target

Use the fixed SLURM wrappers:

- `motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_prompt_gemini_slurm.sh`
- `motivated_reasoning/evaluation/slurm/evaluate_inference_outputs_with_local_model_slurm.sh`

## Results Summary To Produce

Make a compact table per run with:

- final training status
- answer reward trajectory
- empty reasoning rate
- malformed tag rate
- median/mean reasoning words
- correctness/influence results
- motivated-reasoning score
- monitor scores
- verdict: suitable for paper/rebuttal plots, negative-result only, or rerun needed

## Plot Updates

If Qwen results are usable:

1. Add their eval-output paths to the plot-data mapping.
2. Update the relevant plotting/reproducibility script.
3. Regenerate only affected plots first.
4. Compare regenerated plots against backups before replacing paper figures.
5. Record exact commands and source paths in `EXPERIMENT_LOG.md`.

