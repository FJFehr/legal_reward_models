#!/usr/bin/env bash
# Build locally (requires sudo), then copy .sif to the cluster.
set -eux
cd "$(dirname "$0")/.."
singularity build container/llm.sif container/llm.def
