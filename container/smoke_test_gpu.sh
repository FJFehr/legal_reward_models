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

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "Running 2-GPU NCCL all-reduce smoke test..."
    if ! apptainer exec --nv "$SIF" bash -lc 'CUDA_VISIBLE_DEVICES=0,1 python - <<'"'"'PY'"'"'
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    tensor = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2.0
    if float(tensor.item()) != expected:
        raise RuntimeError(
            f"Unexpected all-reduce result on rank {rank}: {tensor.item()} != {expected}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("NCCL 2-GPU all-reduce smoke test passed.")
PY'; then
        echo "NCCL smoke test failed. Try: NCCL_CUMEM_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO" >&2
        exit 1
    fi
else
    echo "Skipping NCCL 2-GPU smoke test (fewer than 2 GPUs visible on host)."
fi
