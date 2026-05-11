#!/bin/bash

# Script to analyze reasonableness evaluation results for constitution_and_response
# Run this after the evaluations are complete

set -e  # Exit on error

echo "========================================="
echo "Running analysis for constitution_and_response reasonableness evaluations"
echo "========================================="
echo ""

# Define experiments
experiments=(
    "now-09_20_201429"
    "later-09_20_201440"
    "safe-09_19_182118"
    "risky-09_19_182001"
)

# Run analysis for each experiment
for exp in "${experiments[@]}"; do
    echo ""
    echo "--------------------------------------"
    echo "Analyzing: $exp"
    echo "--------------------------------------"

    python motivated_reasoning/visualization/study_reasonableness_differences.py \
        "$exp" \
        --output-dir analysis_output/reasonableness

    echo "✓ Completed analysis for $exp"
done

echo ""
echo "========================================="
echo "ANALYSIS COMPLETE!"
echo "========================================="
echo ""
echo "Analysis results saved to:"
echo "  analysis_output/reasonableness/<experiment>/<prompt_type>/constitution_and_response/"
echo ""
echo "Now you can regenerate the plots with:"
echo "  python motivated_reasoning/plotting/plot_reasonableness_proportions_paper.py \\"
echo "    --all-experiments \\"
echo "    --evaluator 3-8b-instruct \\"
echo "    --reasonableness-version simple_reasonable_recommendation_v3"
