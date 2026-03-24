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
| `container/` | Singularity definition files and build/test scripts |

### Read order

1. `AGENTS.md` — rules and priorities
2. The relevant `tasks/*.md` brief
3. `orchestrator_prompt.md` — how tasks are dispatched
4. This file — setup and project overview

## Container

Requires [Singularity](https://docs.sylabs.io/guides/latest/admin-guide/installation.html) (or its successor Apptainer — same commands).

```bash
# Install Singularity (Ubuntu/Debian)
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y singularity-ce
```

### Building

```bash
bash container/build.sh
```

If you don't have root or fakeroot on the cluster, build locally then copy the `.sif` over:

```bash
sudo bash container/build.sh
scp container/llm.sif <user>@<cluster>:<project-path>/container/
```

### Smoke test

```bash
bash container/smoke_test_gpu.sh
```

### Using the container

Run any script with `singularity exec`:

```bash
singularity exec --nv container/llm.sif python scripts/run_vllm_inference.py --model Qwen/Qwen3.5-0.8B --device cuda

# Interactive shell
singularity shell --nv container/llm.sif
```

The `--nv` flag passes through host NVIDIA drivers and GPUs. Bind-mount directories to persist outputs:

```bash
singularity exec --nv --bind ./data:/opt/legal-reward-models/data container/llm.sif python scripts/my_script.py
```

See `container/llm.def` for the full definition. The image uses CUDA 12.8 (`cu128`) by default; switch to CUDA 12.1 (`cu121`) if you hit wheel compatibility issues.

### vLLM inference tuning

`scripts/run_vllm_inference.py` supports additional vLLM engine controls:

- `--tensor-parallel-size` (multi-GPU tensor parallelism)
- `--dtype` (for example: `auto`, `float16`, `bfloat16`)
- `--max-model-len` (reduce KV cache memory pressure)
- `--gpu-memory-utilization` (range `(0, 1]`)
- `--enforce-eager` (faster startup, possible throughput tradeoff)
- `--quantization` (free-form vLLM quantization mode)

Example: fit smaller models on a single 24 GiB GPU by limiting context length.

```bash
singularity exec --nv container/llm.sif python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --prompt "Hello from vLLM"
```

Examples: run tensor parallelism on 3, 4, or 8 GPUs.

```bash
# 3 GPUs
singularity exec --nv container/llm.sif bash -lc \
  'CUDA_VISIBLE_DEVICES=0,1,2 python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --tensor-parallel-size 3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'

# 4 GPUs
singularity exec --nv container/llm.sif bash -lc \
  'CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'

# 8 GPUs
singularity exec --nv container/llm.sif bash -lc \
  'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'
```

Optional: reduce startup latency with eager mode.

```bash
singularity exec --nv container/llm.sif python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

Optional: run a quantized checkpoint when supported by the model.

```bash
singularity exec --nv container/llm.sif python scripts/run_vllm_inference.py \
  --model <quantized-model-id> \
  --device cuda \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --quantization awq
```

## Working in this repo

See `AGENTS.md` for coding conventions and task workflow.
