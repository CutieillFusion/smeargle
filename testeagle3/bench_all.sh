#!/bin/bash
# Submit baseline + EAGLE3 SLURM jobs for every Llama-3.1-8B benchmark.
# Usage: ./bench_all.sh           # instruct variant (default)
#        ./bench_all.sh --base    # base pretrained variant

set -euo pipefail

EXTRA=""
if [[ "${1:-}" == "--base" ]]; then
  EXTRA="--base"
fi

BENCHES=(
  mmlu mmlu_pro agieval_en commonsense_qa winogrande bbh
  arc_challenge trivia_qa squad quac boolq drop
  mmlu_cot gpqa ifeval humaneval mbpp_plus gsm8k math
  apibank bfcl gorilla nexus
)

cd "$(dirname "$0")"

for B in "${BENCHES[@]}"; do
  Q="/data/ai_club/smeargle/datasets/${B}/question.jsonl"
  if [[ ! -s "$Q" ]]; then
    echo "Skipping $B: $Q missing or empty"
    continue
  fi
  echo "Submitting $B..."
  sbatch bench_baseline.sh "$B" $EXTRA
  sbatch bench_eagle3.sh   "$B" $EXTRA
done
