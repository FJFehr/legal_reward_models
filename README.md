# Legal Reward Models

A research pipeline for training and evaluating reward models on unverifiable data using RLAIF and SFT.

## Pipeline overview

1. **Data analysis** — explore and understand source legal datasets
2. **Preference pair construction** — convert data into (chosen, rejected) pairs for reward modelling
3. **Training** — fine-tune models via SFT and RLAIF
4. **Evaluation** — assess reward model quality on unverifiable legal data

## Setup

```bash
uv sync --python 3.12 --managed-python
```

Supported Python versions: `3.12`

## Running tests

```bash
uv run pytest tests/
```

## Repository layout

| Path | Role |
|------|------|
| `scripts/` | Runnable scripts (e.g. vLLM inference) |
| `notebooks/` | Jupyter notebooks for data exploration and EDA |
| `tests/` | Pytest test suite |
| `data/` | Local data directory (git-ignored, kept via `.gitkeep`) |
| `AGENTS.md` | Operating contract for coding agents |
| `tasks/` | Task briefs by type: feature, refactor, debug |
| `orchestrator_prompt.md` | Prompt template for dispatching agent tasks |
| `container/` | Apptainer definition files and build/test scripts |

### Read order

1. `AGENTS.md` — rules and priorities
2. The relevant `tasks/*.md` brief
3. `orchestrator_prompt.md` — how tasks are dispatched
4. This file — setup and project overview

## Container

Requires [Apptainer](https://apptainer.org/docs/admin/main/installation.html) (formerly Singularity).

```bash
# Install Apptainer (Ubuntu/Debian)
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer

# Build GPU image (CUDA 12.8, default)
bash container/build.sh gpu

# Build CPU image (for prototyping without a GPU)
bash container/build.sh cpu

# Smoke-test on a GPU node
bash container/smoke_test_gpu.sh

# Smoke-test CPU image
bash container/smoke_test_cpu.sh
```

### Using the container

Run any script with `apptainer exec`:

```bash
# GPU
apptainer exec --nv container/llm.sif python scripts/run_vllm_inference.py --model Qwen/Qwen3.5-0.8B --device cuda

# CPU
apptainer exec container/llm-cpu.sif python scripts/run_vllm_inference.py --model Qwen/Qwen3.5-0.8B --device cpu

# Interactive shell
apptainer shell --nv container/llm.sif
```

The `--nv` flag passes through host NVIDIA drivers and GPUs — use it with the GPU image, omit it with the CPU image. Bind-mount directories to persist outputs:

```bash
apptainer exec --nv --bind ./data:/opt/legal-reward-models/data container/llm.sif python scripts/my_script.py
```

See `container/llm.def` (GPU) and `container/llm-cpu.def` (CPU) for the full definitions. The GPU image uses CUDA 12.8 (`cu128`) by default; switch to CUDA 12.1 (`cu121`) if you hit wheel compatibility issues.

## Working in this repo

See `AGENTS.md` for coding conventions and task workflow.
