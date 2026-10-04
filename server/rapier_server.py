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
CTX = int(os.environ.get("RAPIER_CTX", "65536"))          # engine kMaxCtx is 16384; leave headroom
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


def render_tools(tools):
    """Qwen official chat-template tool block (must match training format exactly)."""
    lines = ["# Tools", "",
             "You may call one or more functions to assist with the user query.",
             "", "You are provided with function signatures within <tools></tools> XML tags:",
             "<tools>"]
    for t in tools:
        lines.append(json.dumps(t, ensure_ascii=False, separators=(",", ":")))
    lines += ["</tools>", "",
              "For each function call, return a json object with function name and arguments within ",
              "<tool_call>", "</tool_call>", "XML tags:",
              "<tool_call>", "{\"name\": <function-name>, \"arguments\": <args-json-object>}", "</tool_call>",
              "###", ""]
    return "\n".join(lines)


import re as _re
_TOOL_CALL_RE = _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.S)


def parse_tool_calls(text):
    """Extract <tool_call>{...}</tool_call> blocks -> OpenAI tool_calls list (+stripped content)."""
    calls = []
    def _mk(m):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"id": "call_" + str(abs(hash(text)) % 10 ** 10),
                              "type": "function",
                              "function": {"name": name,
                                           "arguments": json.dumps(args, separators=(",", ":"))}})
        except Exception:
            pass
        return ""
    stripped = _TOOL_CALL_RE.sub(_mk, text).strip()
    return calls, stripped


def chat_template(messages):
    messages = list(messages)
    tools = None
    if messages and isinstance(messages[0].get("content"), dict):
        pass  # defensive: OpenAI never sends dict content here
    # tool definitions ride as an extra system block on the FIRST request of a session
    if "_rapier_tools" in (messages[0] if messages else {}):
        tools = messages[0]["_rapier_tools"]
        messages = [dict(m) for m in messages]
        messages[0].pop("_rapier_tools")
    out = ""
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            # OpenAI tool result -> Qwen's <tool_response> wrapper inside a user turn
            out += f"<|im_start|>user\n<tool_response>\n{m.get('content','')}\n</tool_response><|im_end|>\n"
        elif role == "assistant" and m.get("tool_calls"):
            # assistant tool-call turn: content + each call in Qwen's <tool_call> form
            parts = []
            if m.get("content"):
                parts.append(m["content"])
            for c in m["tool_calls"]:
                fn = c.get("function", {})
                parts.append("<tool_call>\n" + json.dumps(
                    {"name": fn.get("name"), "arguments": json.loads(fn.get("arguments") or "{}")},
                    separators=(",", ":")) + "\n</tool_call>")
            out += "<|im_start|>assistant\n" + "\n".join(parts) + "<|im_end|>\n"
        else:
            content = m.get("content") or ""
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if tools:
        block = render_tools(tools)
        if out.startswith("<|im_start|>system\n"):
            end = out.index("<|im_end|>")
            merged = out[:end] + "\n\n" + block + out[end:]
            out = merged
        else:
            out = "<|im_start|>system\n" + block + "<|im_end|>\n" + out
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


def generate(messages, max_tokens, temperature, top_p, on_token=None):
    """Returns (text, raw_text) - raw includes any <tool_call> blocks.
    on_token(piece, in_think) streams decoded pieces as they generate (SSE path)."""
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
    # On a follow-up (delta-only) request the engine still emits the argmax of the LAST INGESTED
    # token - which the previous turn already produced. Skip exactly one stale emission so the
    # stream starts at the genuinely new token.
    skip_first = bool(hist) and shared > 0 and not new_tokens is None and len(new_tokens) >= 0 and primed
    req = {"op": "run", "tokens": new_tokens, "gen": max_tokens + (1 if skip_first else 0),
           "mtp": bool(greedy)}
    engine_send(req)
    out_ids, out_text = [], []
    import random
    rng = random.Random()
    eos_hit = False
    think_depth = 0
    seen = 0  # chars of raw already classified (think markers span pieces)
    while True:
        line = engine_recv()
        if line.get("ok") or line.get("op"):
            continue  # stale ack from a reset; not part of this run's stream
        if line.get("done"):
            # ALWAYS drain to done: an early return here leaves stale lines in the pipe and
            # every later request reads the previous response's leftovers (off-by-one-request).
            hist = ids + out_ids
            primed = True
            break
        if eos_hit:
            continue  # engine stops itself after eos; ignore anything until done
        if "id" not in line:
            raise RuntimeError(f"engine line missing id: {line}")
        if skip_first:
            skip_first = False
            out_ids.append(line["id"])   # still counts toward history (it re-states last turn's tail)
            continue
        tid = line["id"]
        out_ids.append(tid)
        if not greedy and tid != EOS_ID and "top" in line and line["top"]:
            tid = sample(line["top"], temperature, top_p, rng)
        if tid == EOS_ID:
            eos_hit = True
            continue
        piece = _tk.decode([tid])
        out_text.append(piece)
        if on_token is not None:
            seen_new = "".join(out_text)[seen:]
            seen += len(seen_new)
            low = seen_new
            while low:
                if think_depth == 0:
                    i = low.find("<think>")
                    if i < 0:
                        if low:
                            on_token(low, False)
                        low = ""
                    else:
                        if i:
                            on_token(low[:i], False)
                        think_depth = 1
                        low = low[i + 7:]
                else:
                    i = low.find("</think>")
                    if i < 0:
                        if low:
                            on_token(low, True)
                        low = ""
                    else:
                        if i:
                            on_token(low[:i], True)
                        think_depth = 0
                        low = low[i + 8:]
    full = "".join(out_text)
    return full, full





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
        tools = body.get("tools") or None
        if tools:
            messages = [dict(m) for m in messages]
            if messages:
                messages[0]["_rapier_tools"] = tools
        want_stream = bool(body.get("stream"))
        if want_stream and os.environ.get("RAPIER_TRACE"):
            import sys as _s
            print("TRACE: stream request, generating...", file=_s.stderr, flush=True)
        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            def sse(obj):
                try:
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            sse({"id": "rapier", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            def on_tok(piece, in_think):
                if not piece:
                    return
                delta = {"reasoning_content": piece} if in_think else {"content": piece}
                sse({"id": "rapier", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            try:
                with eng_lock:
                    raw, _ = generate(messages, max_tokens, temperature, top_p, on_token=on_tok)
                if os.environ.get("RAPIER_TRACE"):
                    import sys as _s
                    print(f"TRACE: generate returned {len(raw)} chars", file=_s.stderr, flush=True)
            except Exception as e:
                sse({"error": str(e)})
                return
            calls, text = parse_tool_calls(raw) if tools else ([], raw)
            if calls:
                sse({"id": "rapier", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "finish_reason": None,
                                  "delta": {"tool_calls": calls}}]})
                fin = "tool_calls"
            else:
                fin = "stop"
            sse({"id": "rapier", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        try:
            with eng_lock:
                raw, _ = generate(messages, max_tokens, temperature, top_p)
        except Exception as e:
            self._json(500, {"error": str(e)})
            return
        calls, text = parse_tool_calls(raw) if tools else ([], raw)
        n_gen = len(raw) // 4 or 1
        dt = time.time() - t0
        self._json(200, {
            "id": "rapier", "object": "chat.completion", "model": "rapier",
            "choices": [{"index": 0,
                         "finish_reason": "tool_calls" if calls else "stop",
                         "message": ({"role": "assistant", "content": text,
                                      "tool_calls": calls} if calls
                                     else {"role": "assistant", "content": text})}],
            "usage": {"prompt_tokens": 0, "completion_tokens": n_gen,
                      "total_tokens": n_gen},
            "timings": {"generation_ms": round(dt * 1000), "tok_s": round(n_gen / dt, 2) if dt else 0},
        })


if __name__ == "__main__":
    engine_start()
    print(f"rapier server on :{PORT} (engine {ENGINE_BIN})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
