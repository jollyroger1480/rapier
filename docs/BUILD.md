# Build notes

Rapier tracks upstream Strata **v0.1.42**: `git checkout v0.1.40.3` before applying
`patches/*.patch` (they are cut as a diff against that tag).

The exact CMake block (also inline in the README) that adds the `qwythos` target to a
Strata HIP build:

```cmake
if(STRATA_ENABLE_HIP)
  add_executable(qwythos src/program/qwythos.cpp src/program/qwythos_prefill.cu)
  set_source_files_properties(src/program/qwythos.cpp src/program/qwythos_prefill.cu PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwythos PRIVATE strata_kernels strata_prefill strata_hip_runtime)
endif()
```

Validated toolchain: ROCm 10.1 (`/opt/rocm` → `/data/rocm-10.1.0`), `clang++` from
`/opt/rocm/llvm/bin`, `CMAKE_HIP_ARCHITECTURES=gfx1030` (add `gfx1034` for an RX
6500-class second card — `_strata_hip_unvalidated` in `cmake/hip_backend.cmake`
already carries both on 0.1.40.3).

The engine picks the largest visible GPU. With a small second card in the box, pin the
big one: `HIP_VISIBLE_DEVICES=1 ./build-hip/qwythos ...`.

Post-update checks: 0.1.37 → 0.1.40.3 rebase (2026-10-07): 76.6 tok/s, ids bit-identical.
0.1.40.3 → 0.1.41 (2026-10-08) and → 0.1.42 (2026-10-10) rebases: same invariant,
batched GEMM prefill now the default (155-157 tok/s @8k ingest, bit-exact A/B via
QWYTHOS_PF_SEQ=1). `build-hip/test_res` = synthetic parity harness for the MMVQ
residual-epilogue kernels (run on a sacrificial card, not your display GPU).
