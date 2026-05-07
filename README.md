# Smeargle

Speculative decoding research repo. Smeargle is our draft-model architecture; we benchmark it against EAGLE-3 on Llama 3.1 8B Instruct.

## Folders

- **`trainsmeargle/`** — Training code for our Smeargle draft model.
- **`traineagle3/`** — Training code for our re-implementation of EAGLE-3.
- **`testsmeargle/`** — Inference / benchmarking scripts for Smeargle (baseline + speculative runs on Spec-Bench and wiki-long).
- **`testeagle3/`** — Inference / benchmarking scripts for our EAGLE-3 implementation, including sliding-window sweeps.
- **`testeagle3old/`** — Inference / benchmarking scripts against the **previously published pretrained** EAGLE-3 checkpoint (reference baseline).
- **`models/`** — Local copies of base models and trained draft checkpoints (Llama 3.1 8B/70B, Qwen 3 8B, GPT-OSS 20B, Smeargle and EAGLE-3 variants).
- **`datasets/`** — Dataset download / preprocessing scripts and cached splits (ShareGPT, UltraChat 200k, Spec-Bench, wiki-long, perfect-blend).
- **`containers/`** — Apptainer/Singularity images (CUDA dev containers) used for training and evaluation jobs.

> **Note:** `testeagle3old/` targets the prior pretrained EAGLE-3 release. All other `*smeargle*` and `*eagle3*` folders are our own implementations.
