#!/bin/bash
#SBATCH --job-name=REFACTOR
#SBATCH --output=models/%j/train_refactor.out
#SBATCH --error=models/%j/train_refactor.err
#SBATCH --partition=dgxh100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --account=undergrad_research

cd /data/ai_club/smeargle/refactor

SAVEDIR=${1:-$SLURM_JOB_ID}

if [ -d "models/$SAVEDIR" ]; then
  echo "Directory models/$SAVEDIR already exists"
  rm -rf "models/$SAVEDIR"
fi

# Mount your project and use host's uv
singularity exec --nv \
  --bind /data/ai_club/smeargle/refactor:/workspace \
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
      --trainpath /datasets/train_5k.jsonl \
      --testpath /datasets/test_5k.jsonl \
      --epochs 2 \
      --savedir $SAVEDIR
  "