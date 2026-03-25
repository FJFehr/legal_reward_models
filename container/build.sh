#!/usr/bin/env bash
# Build locally (requires sudo), then copy .sif to the cluster.
set -eux
cd "$(dirname "$0")/.."

UV_BACKEND="${1:-cu128}"
case "$UV_BACKEND" in
    cu128|cu121) ;;
    *)
        echo "Unsupported backend: $UV_BACKEND (expected cu128 or cu121)" >&2
        exit 1
        ;;
esac

TMP_DEF="$(mktemp)"
trap 'rm -f "$TMP_DEF"' EXIT

sed "s/export UV_TORCH_BACKEND=cu128/export UV_TORCH_BACKEND=${UV_BACKEND}/g" \
    container/llm.def > "$TMP_DEF"

echo "Building container with UV_TORCH_BACKEND=${UV_BACKEND}"
apptainer build container/llm.sif "$TMP_DEF"
