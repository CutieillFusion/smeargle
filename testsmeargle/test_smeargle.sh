#!/bin/bash
#SBATCH --job-name=SMEARGLE
#SBATCH --output=train_smeargle.out
#SBATCH --error=train_smeargle.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/testsmeargle

[ -f inference_debug.jsonl ] && rm inference_debug.jsonl

# Mount project
singularity exec --nv \
  --bind /data/ai_club/smeargle/testsmeargle:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  --bind ~/.local/bin:/usr/local/bin \
  ../containers/cuda_12.1.0-devel-ubuntu22.04.sif \
  bash -c "cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda-12.1 && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH && \
    uv sync && \
    .venv/bin/python gen_answer_llama_3_1_8b.py \
      --smeargle-model-path /models/1epoch_better_smeargle_llama_3_1_8b_instruct \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/mt_bench \
      --temperature 0.0 \
      --depth 1 \
      --warmup-steps 0 \
      --question-begin 0 \
      --question-end 1 \
      --use_smeargle"