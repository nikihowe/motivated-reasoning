#!/usr/bin/env bash
set -euo pipefail

# Regenerate the quantitative plots used by the paper from existing
# evaluation_output/ and analysis_output/ artifacts.
#
# This intentionally uses the current best motivated-reasoning average plot:
# Risky uses copy_constitution_motivated_reasoning_v3, which has full early
# judge coverage. The older submitted-paper asset used the backup suffix
# copy_constitution_motivated_reasoning_v3_backup_20260402 for Risky.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export MPLBACKEND=Agg
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/nas/ucb/nikihowe/conda/envs/motivated_reasoning_env/bin/python}"

"$PYTHON_BIN" motivated_reasoning/plotting/plot_average_evaluation.py
"$PYTHON_BIN" motivated_reasoning/plotting/plot_reward.py
"$PYTHON_BIN" motivated_reasoning/plotting/four_plot_simple_evaluation.py

# HarmBench refusal plot used in the paper was copied from this old output
# namespace, so reproduce it there as well.
"$PYTHON_BIN" motivated_reasoning/plotting/plot_refusal.py \
  harmbench_cot_tags-08_18_162853 \
  --output-dir "plots/new old"

# Response-over-iteration plots for easy (_v2) and hardcore (non-v2)
# appendix figures.
"$PYTHON_BIN" motivated_reasoning/plotting/plot_simple_evaluation.py risky-09_19_182001
"$PYTHON_BIN" motivated_reasoning/plotting/plot_simple_evaluation.py safe-09_19_182118
"$PYTHON_BIN" motivated_reasoning/plotting/plot_simple_evaluation.py now-09_20_201429
"$PYTHON_BIN" motivated_reasoning/plotting/plot_simple_evaluation.py later-09_20_201440

# Motivated-reasoning score distributions.
"$PYTHON_BIN" motivated_reasoning/plotting/plot_evaluation.py \
  harmbench_cot_tags-08_18_162853 \
  --evaluator gemini-25-flash-lite \
  --prompt-type constitutional_cot \
  --suffix copy_constitution_motivated_reasoning_v3

"$PYTHON_BIN" motivated_reasoning/plotting/plot_evaluation.py \
  risky-09_19_182001 \
  --evaluator gemini-25-flash-lite \
  --prompt-type safe_constitutional_cot_v2 \
  --suffix copy_constitution_motivated_reasoning_v3

"$PYTHON_BIN" motivated_reasoning/plotting/plot_evaluation.py \
  safe-09_19_182118 \
  --evaluator gemini-25-flash-lite \
  --prompt-type risky_constitutional_cot_v2 \
  --suffix copy_constitution_motivated_reasoning_v3

"$PYTHON_BIN" motivated_reasoning/plotting/plot_evaluation.py \
  now-09_20_201429 \
  --evaluator gemini-25-flash-lite \
  --prompt-type later_constitutional_cot_v2 \
  --suffix copy_constitution_motivated_reasoning_v3

"$PYTHON_BIN" motivated_reasoning/plotting/plot_evaluation.py \
  later-09_20_201440 \
  --evaluator gemini-25-flash-lite \
  --prompt-type now_constitutional_cot_v2 \
  --suffix copy_constitution_motivated_reasoning_v3

# CoT monitor degradation figure.
"$PYTHON_BIN" motivated_reasoning/plotting/plot_cot_degradation.py \
  --all-experiments \
  --evaluator 3-8b-instruct \
  --reasonableness-version simple_reasonable_recommendation_v3

echo "Paper plots regenerated under ./plots"
