<p align="center"><img src="docs/banner.jpeg" width="800" alt="Rapier — a rapier of circuit traces pinning a chip-skull flag"></p>

<h1 align="center">⚔️ Rapier</h1>

<p align="center"><b>Fast local decode for Qwen3.5-class hybrid models on AMD RDNA2 — with exact-verify MTP.</b><br>
74 tok/s greedy from a 9B Gated-DeltaNet hybrid on an RX 6950 XT — faster than llama.cpp Vulkan on the same card.<br>
Built at [Buccaneer Salvage](https://github.com/jollyroger1480) · X [@jollyroger1480](https://x.com/jollyroger1480)</p>

---

Rapier is a single-model HIP decode engine for the **qwen35 architecture** (the Qwen3.5/Next hybrid:
Gated-DeltaNet linear-attention layers + GQA every 4th layer + an MTP draft head), built as a program
and two kernel patches inside a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata)
(the MoE engine for Qwen3.8-Flash-Next — itself built on [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp)).

It exists because on RDNA2, a decoder purpose-built for one model beats a general one:

| greedy decode, Qwythos-9B-v2 Q6_K, RX 6950 XT (gfx1030, HIP) | tok/s @ 32 tok | tok/s @ 400 tok |
| --- | ---: | ---: |
| llama.cpp b6731 Vulkan (RADV) | 67.1 | ~67 |
| Strata + Rapier, no MTP | 68.6 | 66.3 |
| **Strata + Rapier, MTP** | **68.9** | **73.9** |

Long-context note: before Rapier's attention rewrite the same workload collapsed to **12.6 tok/s**
at position ~200 (a single-warp GQA kernel); it is now flat across context. Total speedup over the
starting point: **44.8 → 73.9 tok/s (+65%)**.

## What's in the box

- **`src/qwythos.cpp`** (~2.9k lines) — the whole decode engine as one program: Q6_K/Q8_0 dp4a GEMVs
  with a CUDA-graph-captured token, fused rms-norm+Q8_1 quantize, weight-matrix concatenation
  (qkv‖z, α‖β, up‖gate, wq‖wk‖wv — one GEMV launch per group), residual-add epilogues fused into the
  GEMVs, a flash-decode-style **parallel attention kernel** (one block per head), and a
  **2-column batched MTP verify pass** that streams the weights once for both tokens.
- **`patches/0001-fused-gdn-silu-q8.patch`** — the fused Gated-DeltaNet chain: conv+SiLU+L2(+q-scale),
  and step+RMS-norm+SiLU with an in-kernel **Q8_1 epilogue** (the next GEMV's input is quantized
  inside the recurrence kernel). 9 kernel launches per layer → 3.
- **`patches/0002-mmvq-rows-residual.patch`** — rows-per-block as a template knob for the Q6_K MMVQ,
  **residual-epilogue GEMV variants** that fold the residual add and the next norm's
  sum-of-squares partials into the GEMV itself (deletes 64 kernels/token), and the 0.1.39
  multi-column occupancy bump (minBlocksPerSM 4 for ROWS<=2), ported and benchmarked neutral
  on this DRAM-bound path.
- **`bench/RESULTS.md`** — the full measurement log: per-kernel bandwidths, the DRAM-ceiling
  experiments (a plain float4 read kernel measures ~482 GB/s on this card; both engines sit at that
  wall), clock/power forensics, and the bug post-mortems.

### The MTP part is *exact*

The draft head (block 32 + the `nextn.*` tensors from the GGUF) proposes; a batched 2-column verify
pass decides. On acceptance the token stream is **bit-identical to non-MTP greedy decoding** —
verified by 400-token sequence diff. Measured acceptance 51–53% ⇒ ~1.5 tokens per verify round.

Four bugs had to die to get there (full stories in `bench/RESULTS.md`): a pinned-buffer DMA race that
fed the verify pass timing-dependent garbage; the `eh_proj` concat order (embedding first!); the
draft consuming the final-**normed** hidden, not the raw residual; and a KV hole at draft position 0.

## Build

You need: an AMD RDNA2 card (tested: RX 6950 XT, gfx1030; ROCm ≥ 6 with HIPCC), cmake, and a
**qwen35-architecture GGUF** — tested with
[empero-ai/Qwythos-9B-v2](https://huggingface.co/empero-ai/Qwythos-9B-v2) (Q6_K MTP).

```sh
git clone https://github.com/Niko1221/Strata
cd Strata
git apply /path/to/rapier/patches/*.patch
cp /path/to/rapier/src/qwythos.cpp src/program/

# add the target (or copy the block from the patch notes in docs/BUILD.md):
cat >> CMakeLists.txt <<'EOF'
if(STRATA_ENABLE_HIP)
  add_executable(qwythos src/program/qwythos.cpp)
  set_source_files_properties(src/program/qwythos.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwythos PRIVATE strata_kernels strata_hip_runtime)
endif()
EOF

cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip --target qwythos -j8
```

## Run

```sh
# token ids in, token ids out (no tokenizer in the engine; ids match the GGUF's vocab)
build-hip/qwythos --tokens 9707 --n 64
```

Diagnostics (env flags): `QWYTHOS_PROBE=1` per-kernel timings + DRAM-ceiling micro-benches ·
`QWYTHOS_TOPK=1` top-8 logits of the first token · `QWYTHOS_NOMTP=1` A/B without speculation ·
`QWYTHOS_MTPTRACE=1` draft/verify decisions · `QWYTHOS_Q6ROWS=1|2|4|8` GEMV sweep.

## Real-life serving (new)

`--serve` turns the engine into a JSON-lines server on stdin/stdout (reset / run ops, greedy-MTP
bit-exact streams, top-40 logits for sampling, 16K context). `server/rapier_server.py` wraps that
in an OpenAI-compatible chat API (`/v1/chat/completions`, `/health`, `/v1/models`) with the Qwen
tokenizer, ChatML template, temperature/top-p sampling, and an incremental conversation session.

Measured on the RX 6950 XT through the full HTTP stack: **60.8 tok/s greedy-MTP** (GPU argmax,
no logits round-trip), ~46 tok/s sampled. **64K context** (caches preallocated at load), SSE
streaming with `reasoning_content` deltas, and OpenAI **tool calling** end to end. Ingestion is
batched: 8-token chunks on the multi-column GEMVs at **199 tok/s** (3x per-token), so agent-size
system prompts are a one-time cost per session — later turns ingest only the new tokens.

```sh
HIP_VISIBLE_DEVICES=1 python3 server/rapier_server.py     # :8084, OpenAI-compatible
curl http://127.0.0.1:8084/v1/chat/completions -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64,"temperature":0}'
```

`tests/test_rapier_serve.py` verifies: tokenizer ground-truth encode, 40-token bit-exact engine
stream vs the recorded MTP run, and a live server chat round-trip. All green.

## Status & honest limits

A **research prototype benchmarked on one model and one card**: tensor shapes are compile-time
constants for Qwythos-9B (4096 hidden, 32 v-heads, …); no tokenizer, no server, greedy only.
The RDNA2 DRAM ceiling (~0.5 GB/s per GB/s of paper spec, measured) bounds everything — Rapier wins
by removing launches and passes, not by breaking physics. Bugs and measurements welcome.

## Credits & license

- [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT) — the host engine, kernels this builds on.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) — quantization formats and dot-product
  arithmetic adapted in the kernels.
- [empero-ai/Qwythos-9B-v2](https://huggingface.co/empero-ai/Qwythos-9B-v2) — the model (fine-tune of
  Qwen3.5-9B) used for development and benchmarks.

MIT — see [LICENSE](LICENSE). Parts adapted from MIT-licensed Strata/llama.cpp carry their notices
in the source headers.

## Support the yard

Rapier is free, forever. If it made your GPU faster, a coffee keeps the soldering irons hot:

**💵 Cash App: [$jollyroger1480](https://cash.app/$jollyroger1480)**
**🐦 X: [@jollyroger1480](https://x.com/jollyroger1480)**
