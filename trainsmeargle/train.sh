#!/bin/bash
#SBATCH --job-name=SMEARGLE
#SBATCH --output=train_smeargle.out
#SBATCH --error=train_smeargle.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/trainsmeargle

SAVEDIR=${1:-$SLURM_JOB_ID}

# Mount your project and use host's uv
singularity exec --nv \
  --bind /data/ai_club/smeargle/trainsmeargle:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  ../containers/cuda_12.0.0-devel-ubuntu22.04.sif \
  bash -c "cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    .venv/bin/python .venv/bin/deepspeed main.py \
      --deepspeed_config ds_config.json \
      --basepath /models/llama_3_1_8b_instruct \
      --trainpath /datasets/train.jsonl \
      --testpath /datasets/test.jsonl \
      --savedir $SAVEDIR"