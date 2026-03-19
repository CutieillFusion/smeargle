#!/bin/bash
#SBATCH --job-name=SMEARGLE
#SBATCH --output=models/%j/train_smeargle.out
#SBATCH --error=models/%j/train_smeargle.err
#SBATCH --partition=dgxh100
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/trainsmeargle

SAVEDIR=${1:-$SLURM_JOB_ID}

# Mount your project and use host's uv
singularity exec --nv \
  --bind /data/ai_club/smeargle/trainsmeargle:/workspace \
  --bind /data/ai_club/smeargle/datasets:/datasets \
  --bind /data/ai_club/smeargle/models:/models \
  --bind ~/.local/bin:/usr/local/bin \
  ../containers/cuda_12.1.0-devel-ubuntu22.04.sif \
  bash -c "
    cd /workspace && \
    unset VIRTUAL_ENV && \
    export CUDA_HOME=/usr/local/cuda-12.1 && \
    export PATH=\$CUDA_HOME/bin:\$PATH && \
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH && \
    uv sync && \
    .venv/bin/python .venv/bin/deepspeed --master_port 29000 main.py \
      --deepspeed_config ds_config.json \
      --basepath /models/llama_3_1_8b_instruct \
      --trainpath /datasets/train.jsonl \
      --testpath /datasets/test.jsonl \
      --epochs 40 \
      --savedir $SAVEDIR
  "