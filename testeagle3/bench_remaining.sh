#!/bin/bash
# Resubmit only the baseline / EAGLE3 benchmark jobs that did NOT finish.
# "Finished" = output jsonl exists and has as many lines as the dataset's
# question.jsonl (the gen script always writes a row per question, even skipped).
#
# Usage: ./bench_remaining.sh           # instruct variant (default)
#        ./bench_remaining.sh --base    # base pretrained variant

set -euo pipefail

EXTRA=""
if [[ "${1:-}" == "--base" ]]; then
  EXTRA="--base"
fi

# benchmark : which side(s) still need to run (baseline / eagle3 / both)
REMAINING=(
  "mmlu:both"
  "mmlu_pro:both"
  "agieval_en:baseline"
  "winogrande:baseline"
  "bbh:both"
  "arc_challenge:baseline"
  "trivia_qa:baseline"
  "squad:both"
  "quac:baseline"
  "drop:eagle3"
  "mmlu_cot:both"
  "humaneval:baseline"
)

cd "$(dirname "$0")"

for ENTRY in "${REMAINING[@]}"; do
  B="${ENTRY%%:*}"
  WHICH="${ENTRY##*:}"
  Q="/data/ai_club/smeargle/datasets/${B}/question.jsonl"
  if [[ ! -s "$Q" ]]; then
    echo "Skipping $B: $Q missing or empty"
    continue
  fi
  case "$WHICH" in
    baseline)
      echo "Submitting $B (baseline only)..."
      sbatch bench_baseline.sh "$B" $EXTRA
      ;;
    eagle3)
      echo "Submitting $B (eagle3 only)..."
      sbatch bench_eagle3.sh   "$B" $EXTRA
      ;;
    both)
      echo "Submitting $B (baseline + eagle3)..."
      sbatch bench_baseline.sh "$B" $EXTRA
      sbatch bench_eagle3.sh   "$B" $EXTRA
      ;;
    *)
      echo "Unknown target '$WHICH' for $B"; exit 2
      ;;
  esac
done
