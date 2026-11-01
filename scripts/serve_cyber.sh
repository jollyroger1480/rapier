#!/usr/bin/env bash
# Cyber brain: Qwen3-14B-Uncensored Q5_K_M on the 6950 XT via llama.cpp Vulkan.
# The qwythos engine is arch-locked to Qwen3.5-9B-hybrid shapes (compile-time
# constants), so sibling brains ride llama.cpp. Serves the OpenAI API on :8085
# with the 64K window Hermes Agent requires (YaRN rope scaling + q4 KV so the
# whole 64K KV fits VRAM next to the 9.79 GB of weights).
#
#   decode ~46 tok/s, prefill ~129 tok/s (6950 XT, 2026-11-01)
#   Hermes: provider `cyber` / alias `sec` (config.yaml)
set -euo pipefail
MODEL="${CYBER_MODEL:-/data/hermes/models/qwen3-14b-uncensored/Qwen3-14B-Uncensored.Q5_K_M.gguf}"
BUILD="${CYBER_BUILD:-/data/phone-lab/llamacpp-pr/build/bin/llama-server}"
exec "$BUILD" \
  -m "$MODEL" \
  --alias cyber-brain \
  --host 127.0.0.1 --port "${CYBER_PORT:-8085}" \
  -ngl 99 -c 65536 \
  --rope-scaling yarn --rope-scale 2.0 --yarn-orig-ctx 32768 \
  -ctk q4_0 -ctv q4_0 \
  --flash-attn auto --jinja --api-key local
