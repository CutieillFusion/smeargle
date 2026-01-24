#!/bin/bash
#SBATCH --job-name=EAGLE3
#SBATCH --output=train_eagle3.out
#SBATCH --error=train_eagle3.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/testeagle3

# Mount project
singularity exec --nv \
  --bind /data/ai_club/smeargle/testeagle3:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  ../containers/cuda_12.0.0-devel-ubuntu22.04.sif \
  bash -c "cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    .venv/bin/python gen_answer_llama_3_1_8b.py \
      --eagle3-model-path /models/1epoch_eagle3_llama_3_1_8b_instruct \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/mt_bench \
      --temperature 0.0 \
      --top-k 2 \
      --depth 7 \
      --warmup-steps 0 \
      --question-begin 0 \
      --question-end 1 \
      --use_eagle3"