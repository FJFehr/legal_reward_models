#!/usr/bin/env bash
set -eux

SIF="${1:-container/llm.sif}"

# Host GPU check
nvidia-smi

# Container checks
apptainer exec --nv "$SIF" python -V
apptainer exec --nv "$SIF" cmake --version
apptainer exec --nv "$SIF" ninja --version
apptainer exec --nv "$SIF" python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
apptainer exec --nv "$SIF" python -c "import vllm; print('vLLM version:', vllm.__version__)"
apptainer exec --nv "$SIF" python -c "import transformers; print('transformers version:', transformers.__version__)"
