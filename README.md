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
| `.agents/` | Agent operating contract — rules, style, prompt template, and task briefs (see below) |
| `container/` | Apptainer/Singularity definition files and build/test scripts |

## Agentic workflow

The `.agents/` folder is a portable operating contract for coding agents (Claude Code, Copilot, Cursor, or any LLM-based tool). It defines how agents should behave, what style to follow, and how tasks are structured — so you get consistent, reviewable output regardless of which agent runs the work.

### What's inside

| File | Purpose |
|------|---------|
| `AGENTS.md` | The root contract — repo priorities, working rules, version control policy, verification steps, and response format. Every agent reads this first. |
| `STYLE.md` | Coding style guide — naming, formatting, modularity, error handling, tests. |
| `PROMPT_TEMPLATE.md` | A fill-in template for dispatching tasks to an agent. |
| `tasks/feature.md` | Process guide for adding new functionality. |
| `tasks/debug.md` | Process guide for diagnosing and fixing bugs. |
| `tasks/refactor.md` | Process guide for restructuring code without changing behaviour. |

Agents read files in precedence order: `AGENTS.md` > `STYLE.md` > `PROMPT_TEMPLATE.md` > task brief > `README.md`. When instructions conflict, higher-precedence files win.

### How to prompt an agent

Copy the template from `PROMPT_TEMPLATE.md` and fill in the fields:

```
Task type: feature
Mode: implement
Objective: Add a --dry-run flag to the training script that skips writing checkpoints.
Acceptance criteria: Running with --dry-run completes without writing to disk. Existing runs unaffected.
Constraints: Do not change the checkpoint format or add new dependencies.
Known unknowns: Unclear whether eval should also be skipped; assume no.
Relevant files: train.py, tasks/feature.md
Out-of-scope files: data/, tests/ (read only)
Verification: uv run python train.py --dry-run exits cleanly; existing tests pass.
```

The agent will then follow the matching task brief (`tasks/feature.md` in this case) for its process, apply the working rules from `AGENTS.md`, and respond using the structured output format.

**Tips:**
- Set **Mode** to `propose-only` when you want a plan without code changes.
- Use **Constraints** and **Out-of-scope files** to keep the agent focused.
- The **Known unknowns** field lets the agent handle ambiguity explicitly rather than guessing silently.

## Container

Requires [Apptainer](https://apptainer.org/docs/admin/main/installation.html) (the successor to Singularity — same commands, same `.sif` format).

```bash
# Install Apptainer (Ubuntu/Debian)
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer

# Or install in userspace (no root needed)
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - ~/.local/apptainer

# Add to ~/.bashrc so apptainer is always on PATH
echo 'export PATH=~/.local/apptainer/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Building

```bash
# Default backend (CUDA 12.8 wheels)
bash container/build.sh

# Compatibility profile (CUDA 12.1 wheels)
bash container/build.sh cu121
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

Run any script with `apptainer exec`:

```bash
apptainer exec --nv container/llm.sif python scripts/run_vllm_inference.py --model Qwen/Qwen3.5-0.8B --device cuda

# Interactive shell
apptainer shell --nv container/llm.sif
```

The `--nv` flag passes through host NVIDIA drivers and GPUs. Bind-mount directories to persist outputs:

```bash
apptainer exec --nv --bind ./data:/opt/legal-reward-models/data container/llm.sif python scripts/my_script.py
```

See `container/llm.def` for the full definition. The image uses CUDA 12.8 (`cu128`) by default; pass `cu121` to `container/build.sh` if cluster runtime compatibility is better with CUDA 12.1 wheels.

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
apptainer exec --nv container/llm.sif python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --prompt "Hello from vLLM"
```

Tensor-parallel rule: `--tensor-parallel-size` must divide the model's attention head counts (and KV head counts when present). For `Qwen/Qwen3.5-0.8B` (`num_attention_heads=16`), valid values are `1, 2, 4, 8, 16`.

Examples: run tensor parallelism on 2, 4, or 8 GPUs.

```bash
# 2 GPUs
apptainer exec --nv container/llm.sif bash -lc \
  'CUDA_VISIBLE_DEVICES=0,1 python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'

# 4 GPUs
apptainer exec --nv container/llm.sif bash -lc \
  'CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'

# 8 GPUs
apptainer exec --nv container/llm.sif bash -lc \
  'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9'
```

Optional: reduce startup latency with eager mode.

```bash
apptainer exec --nv container/llm.sif python scripts/run_vllm_inference.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cuda \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

Optional: run a quantized checkpoint when supported by the model.

```bash
apptainer exec --nv container/llm.sif python scripts/run_vllm_inference.py \
  --model <quantized-model-id> \
  --device cuda \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --quantization awq
```

### Troubleshooting NCCL init crashes

If multi-GPU startup fails in NCCL initialization (`ncclCommInitRank`/segfault):

1. Run the smoke test first:

```bash
bash container/smoke_test_gpu.sh
```

The smoke script now includes a 2-GPU NCCL all-reduce check when at least two GPUs are visible.

2. Retry inference with conservative NCCL settings:

```bash
apptainer exec --nv container/llm.sif bash -lc \
  'NCCL_CUMEM_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO \
   python scripts/run_vllm_inference.py --model Qwen/Qwen3.5-0.8B --device cuda --tensor-parallel-size 2'
```

3. If `cu128` remains unstable on your cluster nodes, rebuild with the `cu121` profile:

```bash
bash container/build.sh cu121
```

## Working in this repo

See the [Agentic workflow](#agentic-workflow) section above and `.agents/AGENTS.md` for the full operating contract.
