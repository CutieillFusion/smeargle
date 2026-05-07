cd /data/ai_club/smeargle/testeagle3new

# Mount your project and use host's uv
singularity exec --nv \
  --bind /data/ai_club/smeargle/testeagle3new:/workspace \
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
      --eagle3-model-path /models/eagle3_neurips \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/wiki_long \
      --temperature 1.0 \
      --depth 5 \
      --warmup-steps 3 \
      --use_eagle3"

singularity exec --nv \
  --bind /data/ai_club/smeargle/testeagle3new:/workspace \
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
      --eagle3-model-path /models/eagle3_neurips \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/spec_bench \
      --temperature 0.0 \
      --depth 5 \
      --warmup-steps 3 \
      --use_eagle3"

singularity exec --nv \
  --bind /data/ai_club/smeargle/testeagle3new:/workspace \
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
      --eagle3-model-path /models/eagle3_neurips \
      --base-model-path /models/llama_3_1_8b_instruct \
      --answer-file-path /workspace \
      --benchmark-path /datasets/spec_bench \
      --temperature 1.0 \
      --depth 5 \
      --warmup-steps 3 \
      --use_eagle3"