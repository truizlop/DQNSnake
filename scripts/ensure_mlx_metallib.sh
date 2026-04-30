#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_DIR="${ROOT_DIR}/swift"
CHECKOUT_DIR="${SWIFT_DIR}/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
OUTPUT_METALLIB="${ROOT_DIR}/default.metallib"
TMP_BUILD_DIR="${TMPDIR:-/tmp}/dqnsnake-mlx-metallib-build"

if [[ -f "${OUTPUT_METALLIB}" ]]; then
  exit 0
fi

if [[ ! -d "${CHECKOUT_DIR}" ]]; then
  echo "mlx-swift checkout not found. Run swift build once first." >&2
  exit 1
fi

cmake -S "${CHECKOUT_DIR}" -B "${TMP_BUILD_DIR}" \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON=OFF >/dev/null
cmake --build "${TMP_BUILD_DIR}" --target mlx-metallib -j8 >/dev/null

METALLIB_PATH="${TMP_BUILD_DIR}/mlx/backend/metal/kernels/mlx.metallib"
if [[ ! -f "${METALLIB_PATH}" ]]; then
  echo "Failed to build mlx.metallib at expected path: ${METALLIB_PATH}" >&2
  exit 1
fi

cp "${METALLIB_PATH}" "${OUTPUT_METALLIB}"

DEBUG_BUILD_DIR="${SWIFT_DIR}/.build/arm64-apple-macosx/debug"
if [[ -d "${DEBUG_BUILD_DIR}" ]]; then
  cp "${METALLIB_PATH}" "${DEBUG_BUILD_DIR}/mlx.metallib"
fi
