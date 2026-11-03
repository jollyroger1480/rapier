# Build notes

Two sources, pick one:

- **Fork branch (recommended, current)**: `jollyroger1480/Strata` branch
  `port/qwythos-0.2.10` — upstream **v0.2.10** with the whole port as commits
  (qwythos program, kernels, target wiring, and the experimental
  `STRATA_HIP_GFX803` Polaris support). Engine program and CMake targets are
  already wired on the branch; clone, configure, build.
- **Patch path (legacy base)**: clone upstream Niko1221/Strata, `git checkout
  v0.1.44.1`, apply `patches/000{1,2,3}-*.patch` (cut as diffs against that
  tag), copy `src/qwythos.cpp` + `src/test_res.cpp`, add the CMake target block
  from the README. `patches/0004-gfx803-polaris-experimental.patch` is NOT for
  this path — it is a port-branch (v0.2.10 base) commit diff and applies there.

The exact CMake block (also inline in the README) that adds the `qwythos` target to a
Strata HIP build:

```cmake
if(STRATA_ENABLE_HIP)
  add_executable(test_res src/test_res.cpp)
  set_source_files_properties(src/test_res.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(test_res PRIVATE strata_kernels strata_hip_runtime)
  add_executable(qwythos src/program/qwythos.cpp)
  set_source_files_properties(src/program/qwythos.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwythos PRIVATE strata_kernels strata_hip_runtime)
endif()
```

Validated toolchain: ROCm 10.1 (`/opt/rocm` → `/data/rocm-10.1.0`), `clang++` from
`/opt/rocm/llvm/bin`, `CMAKE_HIP_ARCHITECTURES=gfx1030` (add `gfx1034` for an RX
6500-class second card — `_strata_hip_unvalidated` in `cmake/hip_backend.cmake`
already carries gfx1030/gfx1034 plus gfx900/gfx803 on the port branch).

The engine picks the largest visible GPU. With a small second card in the box, pin the
big one: `HIP_VISIBLE_DEVICES=1 ./build-hip/qwythos ...`.

Post-update checks: 0.1.37 → 0.1.40.3 rebase (2026-10-07): 76.6 tok/s, ids bit-identical.
0.1.40.3 → 0.1.41 (2026-10-08) and → 0.1.42 (2026-10-10), 0.2.3 (2026-10-23) and 0.2.8 (2026-10-30) rebases: same invariant,
batched GEMM prefill was the default then (155-157 tok/s @8k ingest). That figure is
not the current path: on 2026-10-15 the hipBLAS chunk was removed because its greedy
ids did not match the per-token path. Long prompts use the 8-wide Q8 GEMV,
with that chunk's K scores read once (`QWYTHOS_BN=16` is the same ids).
0.1.42 → 0.1.44.1 (2026-10-15): patches recut against that tag.
`build-hip/test_res` = synthetic parity harness for the MMVQ
residual-epilogue kernels (run on a sacrificial card, not your display GPU).

## Polaris (gfx803) deployment — the COV4 bridge (2026-10-31)

ROCm 10 still compiles gfx803 fine; the wall is the fatbin **code object version**.
Old runtimes (e.g. the ROCm 5.7 stack on a box that never got new ROCm) reject
COV5/COV6 bundles as `[Unsupported]` — the launch silently no-ops and
`hipGetLastError()` can even return success. Verified on an RX 480 with
`AMD_LOG_LEVEL=4`, which prints the exact per-bundle verdict.

Recipe (build on ROCm 10, run on a ROCm 5.7 runtime):

1. Compile with `CMAKE_HIP_FLAGS="-mcode-object-version=4"` and
   `CMAKE_HIP_ARCHITECTURES=gfx803` (gfx803 already rides the unvalidated list).
   This one flag is the whole trick: COV4 bundles are accepted by old runtimes.
2. `patchelf --replace-needed libamdhip64.so.7 libamdhip64.so.5` (or whichever
   soname the old runtime has).
3. If the binary uses `hipGetDevicePropertiesR0600` (any R0600-era symbol),
   same-length binary patch it to `hipGetDeviceProperties` + NULs, and fix the
   matching `.gnu.version_r` aux entry: name `hip_6.0` → `hip_4.2` AND its
   `vna_hash` → `0x0f062772` (SysV hash of "hip_4.2"). The loader compares the
   hash FIRST — a patched name with the old hash fails with
   `version 'hip_4.2' not found` even though the runtime exports it.
4. Run against the old runtime's own libraries; do not mix the new ROCm's
   hsa-runtime/comgr onto LD_LIBRARY_PATH.

Results on an RX 480 (4 GB, Polaris, wave64, dp4a emulated): single-column MMVQ
matches the reference bitwise for every type/shape; the IL multi-column kernels
show fp32 reorder-level noise only (max abs 0.000977 over all cases, 0
non-finite) because wave64 sums in a different order — same class of difference
as any GPU-to-GPU move. Effective T=1 bandwidth: Q6_K ~166 GB/s, IQ4_XS ~125
GB/s, Q5_K ~105 GB/s, 248k-vocab head ~91 GB/s (~half of Polaris peak, through
emulated dp4a). Benchmarks: `mmvq_il_parity --bench` from a COV4 build.

For MoE models on a 4 GB card that also runs the dense engine, run one server at
a time; a `--n-gpu-layers 99 --override-tensor '.ffn_.*_exps\.weight=CPU'
-ctk q8_0 -ctv q8_0` split put all attention on the card and took an 80B-A3B
IQ1_S from 1.49 to 7.8-9.1 tok/s decode on a 4-core host (6x). Big
`--n-cpu-moe` values on RADV can hit `vk::Queue::submit: ErrorDeviceLost` at
load — fall back to the override-tensor form.
