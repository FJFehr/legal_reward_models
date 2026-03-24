#!/usr/bin/env bash
set -eux
cd "$(dirname "$0")/.."

TARGET="${1:-gpu}"

case "$TARGET" in
    gpu)
        singularity build container/llm.sif container/llm.def
        ;;
    cpu)
        singularity build container/llm-cpu.sif container/llm-cpu.def
        ;;
    *)
        echo "Usage: $0 [gpu|cpu]" >&2
        exit 1
        ;;
esac
