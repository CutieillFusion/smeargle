#!/bin/bash
#SBATCH --job-name=SMEARGLE
#SBATCH --output=test_smeargle.out
#SBATCH --error=test_smeargle.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/testsmeargle

# Mount project
singularity exec --nv \
  --bind /data/ai_club/smeargle/testsmeargle:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  ../containers/cuda_12.0.0-devel-ubuntu22.04.sif \
  bash -c "cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    .venv/bin/python gen_answer_llama_3_1_8b.py \
      --smeargle-model-path /models/smeargle_3_1_8b_instruct_perfect_blend \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/wiki_long \
      --temperature 0.0 \
      --depth 7 \
      --warmup-steps 3 \
      --use-smeargle"