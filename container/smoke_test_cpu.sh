#!/usr/bin/env bash
set -eux

SIF="${1:-container/llm-cpu.sif}"

# Container checks
apptainer exec "$SIF" python -V
apptainer exec "$SIF" cmake --version
apptainer exec "$SIF" ninja --version
apptainer exec "$SIF" python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
apptainer exec "$SIF" python -c "import vllm; print('vLLM version:', vllm.__version__)"
apptainer exec "$SIF" python -c "import transformers; print('transformers version:', transformers.__version__)"
