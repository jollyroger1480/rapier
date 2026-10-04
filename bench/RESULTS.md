# Qwythos-9B-v2 trunk decode — speed pass, 2026-10-02

Big card only (RX 6950 XT, gfx1030, HIP build). Greedy generate, token 9707 seed, 32 tokens,
CUDA-graph captured token. Reference: llama.cpp b6731+ Vulkan (RADV) on the same card and GGUF.

| build                          | ms/token | tok/s | first-gen logit |
| ------------------------------ | -------- | ----- | --------------- |
| qwythos.cpp before this pass   | 22.1     | 44.8  | 13 @ 13.9845    |
| qwythos.cpp after this pass    | 18.3     | 54.6  | 13 @ 13.9927    |
| llama.cpp Vulkan reference     | 14.9     | 67.1  | 13 (top-1)      |

Top-8 next-token order after the pass matches the pre-pass build and llama.cpp exactly
(13, 2947, 356, 10417, 34, 198, 25, 48); logit deltas are fp reduction-order noise.

## What changed

1. `native_mmvq.cu` — Q6_K MMVQ rows-per-block is now a template knob (1/2/4/8, env
   `QWYTHOS_Q6ROWS`, default 4). K=4096 single-row blocks gave each lane only 4 kbx
   iterations; Infinity-Cache-resident mats went 733-806 -> 956-992 GB/s. DRAM-resident
   (lm_head) is unchanged at ~553 GB/s - rows do not move the DRAM wall.
2. New residual-epilogue GEMVs (`native_q6_k_mmvq_res`, `native_q8_0_mmvq_res`): the
   ssm_out / attn_wo / ffn_down GEMVs write `y = acc + res` and their block's sum-of-squares
   partials directly, which deleted the 64 per-token residual-add kernels. Deterministic
   (fixed row ownership, fixed partial slots, fixed reduction tree).
3. `fused_gdn.cu` — `fused_gdn_step_norm_silu` (qwen35 silu variant, optional Q8_1 output)
   and `fused_gdn_conv_l2_qk`. The 9-launch GDN chain per layer became 3 launches.
   IMPORTANT: the 1/sqrt(128) q-scale must stay in the step-norm (applied after the state
   read-out). Scaling q before the recurrence AND in the step-norm epilogue distorted the
   out-norm eps by 128x and flipped near-tied logits.
4. `qwythos.cpp` — fused rms+Q8_1 quantize is now multi-block (apply-only; the float row
   stays pre-norm and serves as the residual, so the 64 res copies are gone); swiglu and the
   sigmoid gate write their Q8_1 outputs directly; same-type projection weights are
   concatenated at load (qkv+z, beta+alpha, up+gate, wq+wk+wv) to cut GEMV launches;
   fill_pos runs once per token; one store_kv2 launch. Kernel count per token: ~930 -> ~600.

## Where the remaining gap to llama.cpp lives

Per-token probe after the pass: GEMV 13.9 ms (at the DRAM wall), Q 0.56, D 0.55, H 0.54,
W 0.25, R 0.17, inter-kernel gap 1.32.

- The card's practical DRAM streaming ceiling measured with a plain float4 read kernel is
  ~482 GB/s (lm_head GEMV: 553; copy256: 474). Forcing mclk to level 3 and perf level high
  changes nothing. Both engines sit at this wall; 6.6 GB of weights per token costs ~12-14 ms
  no matter what.
- In-token GEMVs average ~441-475 GB/s effective vs 553 for a sustained single-mat bench;
  repeated single-mat benches are Infinity-Cache-inflated and must not be used as a target.
- To go past llama.cpp, the levers left are op-count and the MTP layer (block 32), not
  the GEMV kernel.

## Diagnostics kept

- `QWYTHOS_PROBE=1` - per-segment timings plus copy256/read256 DRAM-ceiling benches.
- `QWYTHOS_TOPK=1` - top-8 logits of the first generated token.
- `QWYTHOS_OLDGDN=1` - the pre-fusion GDN chain, for A/B parity checks.
- `QWYTHOS_Q6ROWS=1|2|4|8` - Q6_K rows-per-block sweep.

## Addendum: parallel attention (mod 1 of the boost pass)

Replaced the single-warp gqa kernel (one block, 32 threads, all 16 heads) with `gqa_head_kernel`
(one block per head, 256 threads: warp-per-token scores, block-tree softmax, chunked coalesced
V accumulation). The old kernel made per-token cost grow with context; measured gpu_ms average
over a 400-token generation fell 79.2 -> 15.0 ms. Greedy generation diverges from llama.cpp at
near-tied logprobs (~0.1) after ~17 shared tokens - fp reordering, not math error; first token
and top-8 order still match exactly.

| generate (tokens) | before mods | after mod 1 | llama.cpp Vulkan |
| ----------------- | ----------- | ----------- | ---------------- |
| 32 (pos ~16)      | 53.9 t/s    | **68.6 t/s**| 67.1 t/s         |
| 64 (pos ~32)      | ~44.9 t/s   | **68.4 t/s**| ~67 t/s          |
| 400 (pos ~200)    | 12.6 t/s    | **66.3 t/s**| ~67 t/s (flat)   |

Remaining profile: GEMV 13.9 ms (DRAM wall), attention glue 0.27, GDN chain 0.56, gaps 1.35.
Next lever: MTP speculative decode (block 32) - needs draft-head forward + multi-token batched
verify (ncols=2 GEMV exists in the kernels; attention needs a causal-mask batch variant; GDN
needs per-draft-step state snapshots for rollback). Expected ~1.3-1.8x on accepted drafts.

## Addendum 2: MTP speculative decode (mod 2) — WORKING

Draft = blk.32 (the GGUF's MTP layer) + nextn glue; verify = a 2-column batched pass
(ncols=2 GEMVs stream weights once for both tokens; per-token GDN/attention/norm ops run
per column; GDN states snapshotted after col0 for rollback). Greedy-exact: the MTP token
stream is IDENTICAL to the non-MTP greedy stream (400-token diff clean).

| config                      | tok/s (n=400) | acceptance |
| --------------------------- | ------------- | ---------- |
| non-MTP (mod 1 baseline)    | 66.3-66.6     | -          |
| + MTP                       | **73.9**      | 51-53%     |
| llama.cpp Vulkan reference  | 67.1          | -          |

n=32: 68.9, n=64: 71.1 (acceptance ramps as context grows). Strata now beats llama.cpp
by ~10% at long context with exact greedy parity. Session total: 44.8 -> 73.9 tok/s (+65%).

### The four bugs that had to die first

1. **emb_row DMA race (the big one)**: launch_token_verify dequantized both embedding
   rows into ONE pinned buffer with two async H2D copies - the CPU dequantize for column 1
   overwrote the buffer while column 0's copy was still in flight, so col0's bytes were
   timing-dependent garbage. Fixed with a double-wide staging buffer + a single copy.
   Symptom: verify logits argmax subtly-but-catastrophically wrong, varying run to run.
2. **eh_proj concat order**: llama.cpp's qwen35 graph_mtp concats (enorm(emb), hnorm(h)) -
   embedding FIRST. Ours was reversed; the draft projected scrambled inputs.
3. **Draft input hidden**: the MTP consumes the FINAL-NORMED trunk hidden
   (llama.cpp t_h_nextn = norm(h, output_norm)), not the raw residual. We materialize
   h_normed in the token graph now, and the verify computes normed hiddens for both columns.
4. **Draft KV slot 0**: the draft layer needs position 0 filled with the shifted-hidden
   convention (zero h_-1, emb(prompt token 0)) or every draft step attends a hole.
   Also fixed stale raw-hidden feeds at the post-verify draft call sites.

Diagnostics added: QWYTHOS_CMPV / CMPV_N (one-shot verify-vs-single layer bisection with
stage dumps), QWYTHOS_MTPTRACE, QWYTHOS_NOACCEPT, QWYTHOS_VSKIP1, QWYTHOS_GEMV1COLS.

### 2026-10-03 note: repo moved
/data/projects hit a filesystem quota (363 GB of MV projects) and ENOSPC-truncated the
source mid-write; the tree was recovered from ZCode's workspace_file_before_change
snapshots and moved to /data/hermes/Strata (/data/projects/Strata is a symlink).

## Addendum 3: real-life serving + GPU argmax (2026-10-03)

- `--serve` mode: JSON-lines on stdin (reset/run, 16K ctx, top-40 logits, greedy-MTP
  bit-exact streams). `server/rapier_server.py`: OpenAI-compatible chat on :8084
  (Qwen tokenizer extracted from the GGUF — ground-truth verified — ChatML, temp/top-p
  sampling, incremental conversation session with auto-reset).
- GPU argmax kernel (monotonic-float-bits atomicMax, lowest-index tie-break matching the
  host scan) replaces the per-token 1 MB logits D2H + host scans on the serve path.
- Served through the full HTTP stack: 50.7 → **60.8 tok/s** (greedy-MTP, EOS at 66 tokens).
- Test suite (`tests/test_rapier_serve.py`): tokenizer ground-truth, 40-token engine stream
  bit-exact, server round-trip. All green.
- Wiring shipped: hermes-rapier.service (:8084, conflicts with the other two brains),
  panel switches in service-control + hermes-control, Hermes provider + /fast alias.

## Addendum 4: tool calling (2026-10-03)

OpenAI `tools` now works end-to-end: definitions render in the official Qwen chat-template
format inside the system message, assistant `<tool_call>` output parses back to OpenAI
`tool_calls`, tool results map to `<tool_response>` user turns, and multi-turn assistant
tool-call messages render back correctly. Verified live: get_weather single call, full
loop (call -> result -> "sunny, 24°C, SE 12 km/h" answer), cross-conversation sanity.

The bug that ate the afternoon: `generate()` returned on EOS without draining the engine's
`{"done":true}` line, leaving stale lines in the pipe - every subsequent request read the
previous response's leftovers (one-request-off hallucinations, misattributed 500s).
Fix: always drain to done. Lesson: a 1 ms "generation" is a protocol desync, not a speedup.

## Addendum 5: SSE streaming (2026-10-04)

Hermes (and any OpenAI client) streams; the server only returned one JSON blob, so
streaming clients hung forever. Now: stream:true -> SSE with role chunk, per-token
deltas split into `reasoning_content` (<think> blocks) vs `content`, tool_calls emitted
as a single delta when present, finish chunk + [DONE]. First delta lands immediately;
verified back-to-back requests stable. Context also raised to 64K (Hermes Agent minimum):
all caches preallocated, 14.5/16 GB VRAM, short-ctx speed unchanged.

## Addendum 6: batched prefill (2026-10-04)

Agent sessions prepend huge system prompts; per-token ingestion (weights re-read per
token, ~350 MB) made those minutes-long. Now the serve path ingests in chunks of 8
riding the multi-column GEMVs (weights once per chunk, bitwise per column), GDN
sequential per token (cheap), attention per-token against the cache.

Measured: 2401-token prompt ingested at **199 tok/s** (12 s, was ~36 s per-token),
first generated token at 12 s. Live HTTP: 1188-token prompt + answer in 6.2 s.
Parity: batched-ingest continuation matches the greedy reference stream exactly.

Roadmap: ncols 16, real GEMM prefill (borrow Strata's prefill kernels), capture the
batch as a graph. Each is another multiple.
