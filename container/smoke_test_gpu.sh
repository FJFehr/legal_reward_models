#!/usr/bin/env bash
set -eux

SIF="${1:-container/llm.sif}"

# Host GPU check
nvidia-smi

# Container checks
singularity exec --nv "$SIF" python -V
singularity exec --nv "$SIF" cmake --version
singularity exec --nv "$SIF" ninja --version
singularity exec --nv "$SIF" python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
singularity exec --nv "$SIF" python -c "import vllm; print('vLLM version:', vllm.__version__)"
singularity exec --nv "$SIF" python -c "import transformers; print('transformers version:', transformers.__version__)"
