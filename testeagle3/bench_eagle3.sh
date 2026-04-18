#!/bin/bash
#SBATCH --job-name=EAGLE3_BENCH
#SBATCH --output=bench_eagle3_%x_%j.out
#SBATCH --error=bench_eagle3_%x_%j.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

set -euo pipefail

BENCH="${1:-}"
if [[ -z "$BENCH" ]]; then
  echo "Usage: $0 <bench_name> [--base] [--window N]"
  exit 2
fi
shift

USE_CHAT_TEMPLATE=true
MODEL_DIR="/models/llama_3_1_8b_instruct"
EAGLE_DIR="/models/eagle3_3_1_8b_instruct_perfect_blend"
WINDOW_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      USE_CHAT_TEMPLATE=false
      MODEL_DIR="/models/llama_3_1_8b"
      shift
      ;;
    --window)
      WINDOW_ARG="--draft-kv-window $2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"; exit 2
      ;;
  esac
done

# Sanity check that base weights exist when --base was passed.
if [[ "$USE_CHAT_TEMPLATE" == "false" ]]; then
  if [[ ! -d "/data/ai_club/smeargle/models/llama_3_1_8b" ]]; then
    echo "ERROR: Base model not present at /data/ai_club/smeargle/models/llama_3_1_8b."
    echo "Download with: huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /data/ai_club/smeargle/models/llama_3_1_8b"
    exit 3
  fi
fi

cd /data/ai_club/smeargle/testeagle3

singularity exec --nv \
  --bind /data/ai_club/smeargle/testeagle3:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  ../containers/cuda_12.0.0-devel-ubuntu22.04.sif \
  bash -c "cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    .venv/bin/python gen_answer_llama_3_1_8b_bench.py \
      --eagle3-model-path ${EAGLE_DIR} \
      --base-model-path ${MODEL_DIR} \
      --answer-file-path /workspace \
      --benchmark-path /datasets/${BENCH} \
      --temperature 0.0 \
      --depth 5 \
      --warmup-steps 3 \
      --use-chat-template ${USE_CHAT_TEMPLATE} \
      --max-prompt-tokens 8000 \
      --use_eagle3 ${WINDOW_ARG}"
