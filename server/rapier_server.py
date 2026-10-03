#!/usr/bin/env python3
"""Rapier server — OpenAI-compatible chat API over the qwythos HIP engine.

Speaks JSON lines to the engine on stdin (see qwythos.cpp --serve):
  {"op":"reset"}
  {"op":"run","tokens":[..],"gen":N,"mtp":B}
  engine replies with {"id":..,"top":[[id,logit]..]} per token + {"done":true}

Owns: tokenization (HF tokenizers, Qwen BPE), chat template (Qwen ChatML),
sampling (temperature / top_p over the engine's top-40 logits), the conversation
session (incremental ingestion; reset when the context fills), and the HTTP layer.

Endpoints: /health  /v1/models  /v1/chat/completions (non-streaming)
"""
import json
import subprocess
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOME = os.path.expanduser("~")
ENGINE_BIN = os.environ.get("RAPIER_ENGINE", "/data/hermes/Strata/build-hip/qwythos")
MODEL_FILE = os.environ.get("RAPIER_MODEL", "/data/hermes/models/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q6_K.gguf")
TOK_DIR = os.environ.get("RAPIER_TOKENIZER", os.path.join(os.path.dirname(__file__), "tokenizer"))
PORT = int(os.environ.get("RAPIER_PORT", "8084"))
API_KEY = os.environ.get("RAPIER_API_KEY", "local")
CTX = int(os.environ.get("RAPIER_CTX", "12288"))          # engine kMaxCtx is 16384; leave headroom
EOS_ID = 248046
TOPK = 40

from tokenizers import Tokenizer
_tk = Tokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json"))

eng = None          # subprocess handle
eng_lock = threading.Lock()
hist = []           # ingested token ids (the engine's context)
primed = False


def engine_start():
    global eng
    if eng is not None and eng.poll() is None:
        return
    eng = subprocess.Popen(
        [ENGINE_BIN, "--model", MODEL_FILE, "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=sys.stderr, text=True, bufsize=1,
        env={**os.environ, "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "1")},
    )


def engine_send(obj):
    eng.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
    eng.stdin.flush()


def engine_recv():
    while True:
        line = eng.stdout.readline()
        if not line:
            raise RuntimeError("engine died")
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except Exception:
                raise RuntimeError(f"engine sent non-JSON: {line[:200]}")
            # debug: surface the parsed shape on missing keys


def engine_reset():
    global hist, primed
    engine_send({"op": "reset"})
    engine_recv()
    hist = []
    primed = False


def chat_template(messages):
    out = ""
    for m in messages:
        out += f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out


def sample(top, temperature, top_p, rng):
    """top = [[id, logit]..]; renormalize with temperature, apply nucleus, sample."""
    import math
    mx = max(l for _, l in top)
    probs = [(i, math.exp((l - mx) / max(temperature, 1e-5))) for i, l in top]
    total = sum(p for _, p in probs)
    probs = [(i, p / total) for i, p in probs]
    if top_p < 1.0:
        kept, acc = [], 0.0
        for i, p in probs:                       # already sorted by logit desc
            kept.append((i, p))
            acc += p
            if acc >= top_p:
                break
        probs = kept
        total = sum(p for _, p in probs)
        probs = [(i, p / total) for i, p in probs]
    r = rng.random()
    acc = 0.0
    for i, p in probs:
        acc += p
        if r <= acc:
            return i
    return probs[-1][0]


def generate(messages, max_tokens, temperature, top_p):
    global hist, primed
    engine_start()
    ids = _tk.encode(chat_template(messages)).ids
    # find the longest shared prefix with the already-ingested history
    shared = 0
    for a, b in zip(hist, ids):
        if a != b:
            break
        shared += 1
    new_tokens = ids[shared:]
    if not new_tokens and not primed:
        new_tokens = ids            # engine has nothing yet
    if len(hist) + len(new_tokens) + max_tokens + 16 > CTX or shared == 0:
        engine_reset()              # context full or conversation changed: start over
        new_tokens = ids
    greedy = (temperature or 0) < 1e-5
    req = {"op": "run", "tokens": new_tokens, "gen": max_tokens, "mtp": bool(greedy)}
    engine_send(req)
    out_ids, out_text = [], []
    import random
    rng = random.Random()
    while True:
        line = engine_recv()
        if line.get("done"):
            hist = ids + out_ids
            primed = True
            break
        if "id" not in line:
            raise RuntimeError(f"engine line missing id: {line}")
        tid = line["id"]
        out_ids.append(tid)
        if not greedy and tid != EOS_ID and "top" in line and line["top"]:
            tid = sample(line["top"], temperature, top_p, rng)
        if tid == EOS_ID:
            break
        piece = _tk.decode([tid])
        out_text.append(piece)
    return out_text





class Handler(BaseHTTPRequestHandler):
    def _auth(self):
        return API_KEY == "local" or self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ok = eng is not None and eng.poll() is None
            self._json(200, {"status": "ok" if ok else "starting"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "rapier", "object": "model", "owned_by": "buccaneer"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        if not self._auth():
            self._json(401, {"error": "bad key"})
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.6))
        top_p = float(body.get("top_p", 0.95))
        t0 = time.time()
        try:
            with eng_lock:
                pieces = generate(messages, max_tokens, temperature, top_p)
        except Exception as e:
            self._json(500, {"error": str(e)})
            return
        text = "".join(pieces)
        n_gen = len(pieces)
        dt = time.time() - t0
        self._json(200, {
            "id": "rapier", "object": "chat.completion", "model": "rapier",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": n_gen,
                      "total_tokens": n_gen},
            "timings": {"generation_ms": round(dt * 1000), "tok_s": round(n_gen / dt, 2) if dt else 0},
        })


if __name__ == "__main__":
    engine_start()
    print(f"rapier server on :{PORT} (engine {ENGINE_BIN})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
