# Build notes

The exact CMake block (also inline in the README) that adds the `qwythos` target to a
Strata HIP build:

```cmake
if(STRATA_ENABLE_HIP)
  add_executable(qwythos src/program/qwythos.cpp)
  set_source_files_properties(src/program/qwythos.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwythos PRIVATE strata_kernels strata_hip_runtime)
endif()
```

Validated toolchain: ROCm 7.1 (`/opt/rocm`), `clang++` from `/opt/rocm/llvm/bin`,
`CMAKE_HIP_ARCHITECTURES=gfx1030` (add `gfx1034` for an RX 6500-class second card —
also add it to `_strata_hip_unvalidated` in `cmake/hip_backend.cmake`).

The engine picks the largest visible GPU. With a small second card in the box, pin the
big one: `HIP_VISIBLE_DEVICES=1 ./build-hip/qwythos ...`.
