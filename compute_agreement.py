#!/usr/bin/env python3
"""
Compute final agreement statistics for rebuttal.
"""

import json
from pathlib import Path


def compute_full_statistics(data):
    """Compute comprehensive agreement statistics."""
    rated = [ex for ex in data if ex['human_score'] is not None]

    if len(rated) == 0:
        print("❌ No rated examples found")
        return

    # Exact agreement
    exact_match = sum(1 for ex in rated if ex['human_score'] == ex['gemini_score'])

    # Within 1 point
    close_match = sum(1 for ex in rated if abs(ex['human_score'] - ex['gemini_score']) <= 1)

    # Binary classification: 1-2 (genuine) vs 3-5 (motivated)
    def is_motivated(score):
        return score >= 3

    # Confusion matrix
    tp = sum(1 for ex in rated if is_motivated(ex['human_score']) and is_motivated(ex['gemini_score']))
    fp = sum(1 for ex in rated if not is_motivated(ex['human_score']) and is_motivated(ex['gemini_score']))
    tn = sum(1 for ex in rated if not is_motivated(ex['human_score']) and not is_motivated(ex['gemini_score']))
    fn = sum(1 for ex in rated if is_motivated(ex['human_score']) and not is_motivated(ex['gemini_score']))

    binary_match = tp + tn
    p_observed = binary_match / len(rated)

    # Cohen's Kappa
    p_expected = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (len(rated) ** 2)
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected != 1 else 1.0

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Breakdown by Gemini score
    by_gemini_score = {}
    for ex in rated:
        g_score = ex['gemini_score']
        if g_score not in by_gemini_score:
            by_gemini_score[g_score] = []
        by_gemini_score[g_score].append(ex)

    print("\n" + "=" * 80)
    print("📊 FINAL AGREEMENT STATISTICS")
    print("=" * 80)
    print(f"\nTotal examples rated: {len(rated)}/{len(data)}")

    print(f"\n{'─' * 80}")
    print("OVERALL AGREEMENT")
    print(f"{'─' * 80}")
    print(f"Exact match (same score):       {exact_match:2}/{len(rated)} ({100*exact_match/len(rated):.1f}%)")
    print(f"Within 1 point:                 {close_match:2}/{len(rated)} ({100*close_match/len(rated):.1f}%)")
    print(f"Binary agreement (1-2 vs 3-5):  {binary_match:2}/{len(rated)} ({100*binary_match/len(rated):.1f}%)")
    print(f"Cohen's Kappa (binary):         {kappa:.3f}")

    if kappa > 0.8:
        interp = "Almost perfect agreement"
    elif kappa > 0.6:
        interp = "Substantial agreement"
    elif kappa > 0.4:
        interp = "Moderate agreement"
    elif kappa > 0.2:
        interp = "Fair agreement"
    else:
        interp = "Slight/Poor agreement"
    print(f"                                ({interp})")

    print(f"\n{'─' * 80}")
    print("BINARY CLASSIFICATION METRICS (Genuine = 1-2, Motivated = 3-5)")
    print(f"{'─' * 80}")
    print(f"Precision (Gemini says motivated → Human agrees): {precision:.3f}")
    print(f"Recall    (Human says motivated → Gemini caught it): {recall:.3f}")
    print(f"F1 Score:                                           {f1:.3f}")

    print(f"\n{'─' * 80}")
    print("BINARY CONFUSION MATRIX (Genuine 1-2 vs Motivated 3-5)")
    print(f"{'─' * 80}")
    print(f"                      Gemini: Genuine (1-2)    Gemini: Motivated (3-5)")
    print(f"Human: Genuine (1-2)         {tn:2} (TN)                  {fp:2} (FP)")
    print(f"Human: Motivated (3-5)       {fn:2} (FN)                  {tp:2} (TP)")

    # Full 5x5 confusion matrix
    print(f"\n{'─' * 80}")
    print("FULL CONFUSION MATRIX (1-5 scale)")
    print(f"{'─' * 80}")

    # Build the confusion matrix
    confusion = {}
    for h_score in range(1, 6):
        confusion[h_score] = {}
        for g_score in range(1, 6):
            confusion[h_score][g_score] = 0

    for ex in rated:
        h = ex['human_score']
        g = ex['gemini_score']
        if h is not None and g is not None:
            confusion[h][g] = confusion[h][g] + 1

    # Print header
    print("           Gemini Score")
    print("Human      1    2    3    4    5")
    print("Score  " + "─" * 30)

    # Print each row
    for h_score in range(1, 6):
        row = f"  {h_score}    "
        for g_score in range(1, 6):
            count = confusion[h_score][g_score]
            row += f"{count:3}  "
        print(row)

    # Print diagonal (perfect agreement)
    diagonal_sum = sum(confusion[i][i] for i in range(1, 6))
    print(f"\nDiagonal (exact agreement): {diagonal_sum}/{len(rated)} ({100*diagonal_sum/len(rated):.1f}%)")

    print(f"\n{'─' * 80}")
    print("BREAKDOWN BY GEMINI SCORE")
    print(f"{'─' * 80}")
    for g_score in sorted(by_gemini_score.keys()):
        examples = by_gemini_score[g_score]
        h_scores = [ex['human_score'] for ex in examples]
        avg_h_score = sum(h_scores) / len(h_scores)
        agreement = sum(1 for ex in examples if ex['human_score'] == ex['gemini_score'])

        print(f"\nGemini Score {g_score}: {len(examples)} examples")
        print(f"  Human scores: {sorted(h_scores)}")
        print(f"  Mean human score: {avg_h_score:.2f}")
        print(f"  Exact agreement: {agreement}/{len(examples)} ({100*agreement/len(examples):.1f}%)")

    # Find disagreements
    disagreements = [ex for ex in rated if abs(ex['human_score'] - ex['gemini_score']) >= 2]
    if disagreements:
        print(f"\n{'─' * 80}")
        print(f"LARGE DISAGREEMENTS (|Human - Gemini| >= 2): {len(disagreements)}")
        print(f"{'─' * 80}")
        for ex in disagreements:
            print(f"\nExample {ex['example_index']}: Human={ex['human_score']}, Gemini={ex['gemini_score']}")
            print(f"  Answer: {ex['final_answer']}")
            if ex['human_notes']:
                print(f"  Notes: {ex['human_notes']}")

    print("\n" + "=" * 80)
    print("FOR REBUTTAL")
    print("=" * 80)
    print(f"""
Key statistics to report:

1. Sample size: We hand-checked {len(rated)} examples via uniform random sampling
   from two training iterations (early and late).

2. Agreement: Binary agreement (genuine 1-2 vs motivated 3-5) was {100*binary_match/len(rated):.1f}%
   with Cohen's κ = {kappa:.3f} ({interp.lower()}).

3. Precision: When Gemini flagged motivated reasoning, human raters agreed
   {100*precision:.1f}% of the time (precision = {precision:.3f}).

4. Recall: Gemini caught {100*recall:.1f}% of cases where humans identified
   motivated reasoning (recall = {recall:.3f}).

This validates that Gemini 2.5 Flash-Lite is a reliable judge for motivated reasoning
in this setting.
    """)

    return {
        'total_rated': len(rated),
        'exact_agreement': 100 * exact_match / len(rated),
        'binary_agreement': 100 * binary_match / len(rated),
        'cohens_kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    }


def main():
    import sys

    # Use NOW→LATER dataset by default
    if '--comprehensive' in sys.argv:
        data_file = Path('/nas/ucb/nikihowe/motivated-reasoning/hand_check_comprehensive.json')
    elif '--iter8-only' in sys.argv:
        data_file = Path('/nas/ucb/nikihowe/motivated-reasoning/hand_check_iteration8.json')
    elif '--later-only' in sys.argv:
        data_file = Path('/nas/ucb/nikihowe/motivated-reasoning/hand_check_later_only.json')
    else:
        data_file = Path('/nas/ucb/nikihowe/motivated-reasoning/hand_check_now_later.json')

    if not data_file.exists():
        print("❌ Data file not found")
        return

    with open(data_file, 'r') as f:
        data = json.load(f)

    stats = compute_full_statistics(data)

    # Save stats to file
    if stats:
        stats_file = data_file.with_suffix('.stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Statistics saved to: {stats_file}")


if __name__ == '__main__':
    main()
