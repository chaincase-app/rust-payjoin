#!/usr/bin/env bash
set -e

# https://github.com/taiki-e/cargo-llvm-cov?tab=readme-ov-file#merge-coverages-generated-under-different-test-conditions
cargo llvm-cov clean --workspace # remove artifacts that may affect the coverage results
cargo llvm-cov --no-report # v1 configuration
cargo llvm-cov --no-report --features=v2,_danger-local-https,io
cargo llvm-cov report --lcov --output-path lcov.info # generate report without tests
