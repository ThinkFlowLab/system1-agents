#!/usr/bin/env bash
# Build system1-omni's CUDA library (sm_86, A40), the Open-Jev native worker and the Rust frontend.
set -euo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
export PATH=$HOME/.cargo/bin:$PATH
cd ~/kit/system1-omni
rustc --version; nvcc --version | tail -1
src/backends/cuda/qwen3_5/build.sh target/release 86
cargo build --release --locked -q -p omni-open-jev-native -p omni-jev
ls -la target/release/omni-open-jev-native target/release/omni-jev target/release/libqwen3_5_cuda.so
echo "== BUILD_DONE"
