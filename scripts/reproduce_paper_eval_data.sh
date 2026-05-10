#!/usr/bin/env bash
set -euo pipefail

# Print, or optionally run, the exact evaluation/analysis commands needed to
# recreate the evaluation_output/ and analysis_output/ data used by paper plots.
#
# Default is dry-run. To run commands, pass --execute.
# This script does not regenerate model inference outputs; it assumes the
# corresponding inference_output/<run>/<prompt>/iteration-* JSONL files exist.
#
# Iteration policy: paper plots use the baseline plus the first 10 RL training
# iterations. Internally those are iteration-base and iteration-0 through
# iteration-9. Do not submit evals for later iterations unless the paper changes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      MODE="execute"
      shift
      ;;
    *)
      echo "Usage: $0 [--execute]"
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-/nas/ucb/nikihowe/conda/envs/motivated_reasoning_env/bin/python}"
GEMINI_SCRIPT="motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_prompt_gemini.py"
LOCAL_SCRIPT="motivated_reasoning/evaluation/local/evaluate_inference_outputs_with_local_model.py"
ANALYSIS_SCRIPT="motivated_reasoning/visualization/study_reasonableness_differences.py"

BASELINE_AND_TEN_RL=(base 0 1 2 3 4 5 6 7 8 9)
TEN_RL_ONLY=(0 1 2 3 4 5 6 7 8 9)

opposed_runs=(
  "risky-09_19_182001:safe_constitutional_cot_v2"
  "safe-09_19_182118:risky_constitutional_cot_v2"
  "now-09_20_201429:later_constitutional_cot_v2"
  "later-09_20_201440:now_constitutional_cot_v2"
)

hardcore_runs=(
  "risky-09_19_182001:safe_constitutional_cot"
  "safe-09_19_182118:risky_constitutional_cot"
  "now-09_20_201429:later_constitutional_cot"
  "later-09_20_201440:now_constitutional_cot"
)

run_cmd() {
  printf '%q ' "$@"
  echo
  if [[ "$MODE" == "execute" ]]; then
    "$@"
  fi
}

run_gemini_eval() {
  local run_name="$1"
  local inference_prompt_dir="$2"
  local eval_prompt_dir="$3"
  local eval_target="$4"
  local iteration="$5"

  if [[ ! -d "inference_output/$run_name/$inference_prompt_dir/iteration-$iteration" ]]; then
    echo "# Missing inference_output/$run_name/$inference_prompt_dir/iteration-$iteration; skipping"
    return
  fi

  run_cmd "$PYTHON_BIN" "$GEMINI_SCRIPT" \
    --run_name "$run_name" \
    --iteration "$iteration" \
    --inference_prompt_dir "$inference_prompt_dir" \
    --eval_prompt_dir "$eval_prompt_dir" \
    --eval_target "$eval_target" \
    --evaluator flash-lite
}

run_local_monitor_eval() {
  local run_name="$1"
  local inference_prompt_dir="$2"
  local eval_target="$3"
  local iteration="$4"

  if [[ ! -d "inference_output/$run_name/$inference_prompt_dir/iteration-$iteration" ]]; then
    echo "# Missing inference_output/$run_name/$inference_prompt_dir/iteration-$iteration; skipping"
    return
  fi

  run_cmd "$PYTHON_BIN" "$LOCAL_SCRIPT" \
    --run_name "$run_name" \
    --iteration "$iteration" \
    --inference_prompt_dir "$inference_prompt_dir" \
    --eval_prompt_dir simple_reasonable_recommendation_v3 \
    --eval_target "$eval_target" \
    --evaluator_model meta-llama/Meta-Llama-3-8B-Instruct \
    --max_tokens 512
}

echo "# Mode: $MODE"
echo "# Iterations: baseline plus RL iterations 0-9 only"

echo
echo "# Motivated-reasoning score data for score distributions and average plot"
for iteration in "${BASELINE_AND_TEN_RL[@]}"; do
  run_gemini_eval \
    harmbench_cot_tags-08_18_162853 \
    constitutional_cot \
    copy_constitution_motivated_reasoning_v3 \
    constitution_and_reasoning \
    "$iteration"
done

for config in "${opposed_runs[@]}"; do
  IFS=':' read -r run_name prompt_dir <<< "$config"
  for iteration in "${BASELINE_AND_TEN_RL[@]}"; do
    run_gemini_eval \
      "$run_name" \
      "$prompt_dir" \
      copy_constitution_motivated_reasoning_v3 \
      constitution_and_reasoning \
      "$iteration"
  done
done

echo
echo "# Response correctness data for easy/four-way plots and monitor analysis"
for config in "${opposed_runs[@]}"; do
  IFS=':' read -r run_name prompt_dir <<< "$config"
  for iteration in "${BASELINE_AND_TEN_RL[@]}"; do
    run_gemini_eval \
      "$run_name" \
      "$prompt_dir" \
      simple_correct_choice \
      response \
      "$iteration"
  done
done

echo
echo "# Appendix hardcore/original-system-prompt response correctness data"
for config in "${hardcore_runs[@]}"; do
  IFS=':' read -r run_name prompt_dir <<< "$config"
  for iteration in "${BASELINE_AND_TEN_RL[@]}"; do
    run_gemini_eval \
      "$run_name" \
      "$prompt_dir" \
      simple_correct_choice \
      response \
      "$iteration"
  done
done

echo
echo "# HarmBench refusal data"
for iteration in "${BASELINE_AND_TEN_RL[@]}"; do
  run_gemini_eval \
    harmbench_cot_tags-08_18_162853 \
    constitutional_cot \
    simple_compliance \
    response \
    "$iteration"
done

echo
echo "# CoT monitor reasonableness data; no baseline needed for this plot"
for config in "${opposed_runs[@]}"; do
  IFS=':' read -r run_name prompt_dir <<< "$config"
  for eval_target in constitution_and_response constitution_and_reasoning_and_response; do
    for iteration in "${TEN_RL_ONLY[@]}"; do
      run_local_monitor_eval "$run_name" "$prompt_dir" "$eval_target" "$iteration"
    done
  done
done

echo
echo "# analysis_output/reasonableness, after all monitor and correctness evals finish"
for config in "${opposed_runs[@]}"; do
  IFS=':' read -r run_name prompt_dir <<< "$config"
  for iteration in "${TEN_RL_ONLY[@]}"; do
    run_cmd "$PYTHON_BIN" "$ANALYSIS_SCRIPT" \
      "$run_name" \
      --output-dir analysis_output/reasonableness \
      --prompt-type "$prompt_dir" \
      --evaluator 3-8b-instruct \
      --iteration "$iteration" \
      --eval-prompt-dir simple_correct_choice
  done
done

echo
echo "# After these complete, run: ./scripts/reproduce_paper_plots.sh"
