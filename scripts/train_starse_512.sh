#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared StaRSE pair dataset directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
INIT_DIR="${OUTPUT_ROOT}/initialization"
CONTINUOUS_DIR="${OUTPUT_ROOT}/continuous"
POST_TRAINING_DIR="${OUTPUT_ROOT}/post-training"
FINAL_DIR="${OUTPUT_ROOT}/starse"

starse-init \
  --base-model ai-forever/ruBert-base \
  --target-dim 512 \
  --sif-a 5e-5 \
  --zipf-exponent 1.0 \
  --output-dir "${INIT_DIR}"

starse-train \
  --config "${ROOT}/configs/contrastive.yaml" \
  --data-root "${DATA_ROOT}" \
  --model "${INIT_DIR}" \
  --output-dir "${CONTINUOUS_DIR}"

starse-train \
  --config "${ROOT}/configs/sign-projected.yaml" \
  --data-root "${DATA_ROOT}" \
  --model "${CONTINUOUS_DIR}" \
  --output-dir "${POST_TRAINING_DIR}"

starse-project \
  --model "${POST_TRAINING_DIR}" \
  --output-dir "${FINAL_DIR}"

printf 'Final sign-projected model: %s\n' "${FINAL_DIR}"
