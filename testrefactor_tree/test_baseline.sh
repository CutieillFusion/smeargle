#!/bin/bash
#SBATCH --job-name=BASELINE
#SBATCH --output=test_baseline.out
#SBATCH --error=test_baseline.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/testrefactor

# Mount project
singularity exec --nv \
  --bind /data/ai_club/smeargle/testrefactor:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  ../containers/cuda_12.0.0-devel-ubuntu22.04.sif \
  bash -c "cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    .venv/bin/python gen_answer_llama_3_1_8b.py \
      --smeargle-model-path /models/smeargle_3_1_8b_instruct_10_epoch \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/spec_bench \
      --temperature 0.0 \
      --warmup-steps 3 \
      --question-begin 0 \
      --question-end 1"