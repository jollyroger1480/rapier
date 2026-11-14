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
MODEL_FILE = os.environ.get("RAPIER_MODEL", "/data/hermes/models/huihui-qwythos-9b-mythos/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-Q6_K.gguf")
TOK_DIR = os.environ.get("RAPIER_TOKENIZER", os.path.join(os.path.dirname(__file__), "tokenizer"))
PORT = int(os.environ.get("RAPIER_PORT", "8084"))
API_KEY = os.environ.get("RAPIER_API_KEY", "local")
CTX = int(os.environ.get("RAPIER_CTX", "65536"))
PROMPT_BUDGET = int(os.environ.get("RAPIER_PROMPT_BUDGET", "52000"))
GEN_CAP = int(os.environ.get("RAPIER_MAX_GEN", "1024"))
DEADLINE = float(os.environ.get("RAPIER_DEADLINE", "600"))
EOS_ID = 248046
TOPK = 40
# Eager schemas for this 9B only. Grok keeps the full catalog.
# eBay and the rest stay reachable through the search bridges.
KEEP_EAGER = frozenset({
    "terminal", "read_file", "write_file", "patch", "search_files",
    "vision_analyze", "hermes_tool_search", "tool_search",
    "tool_describe", "tool_call",
})


def slim_tools(tools):
    """Drop fat schemas before they hit the 9B. Leave the search bridges."""
    if not tools:
        return tools
    kept = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if fn.get("name") in KEEP_EAGER:
            kept.append(t)
    return kept


_VISION_CACHE = {}
_VISION_PY = "/data/userhome/.hermes/installs/ec7767df1266bc1b/environments/306ed2dac7e54f82a6e7911d0c6b7ed6/venv/bin/python"
_VISION_DROP = (
    "switch", "coil", "sensor", "ignition", "possibly", "appears", "likely",
    "mounting", "voltage", "object", "hardware", "body",
)


def _sanitize_vision(text):
    """Keep a transcription. Drop a guessed product type."""
    if not text:
        return ""
    kept = []
    for line in text.splitlines():
        low = line.lower()
        if any(w in low for w in _VISION_DROP):
            continue
        line = line.strip(" *-")
        if line:
            kept.append(line)
    quote = "\n".join(kept)[:600].strip()
    if not quote or quote.upper() == "UNREADABLE":
        return ""
    return quote


def _vision_quote(path):
    if path in _VISION_CACHE:
        return _VISION_CACHE[path]
    prompt = (
        "Transcribe only characters you can read on labels and stamps. "
        "One token per line. If unclear write UNREADABLE. "
        "Do not describe the object. Do not name what it is."
    )
    script = (
        "import asyncio,json,os,sys\n"
        "from pathlib import Path\n"
        "p=Path('/home/jollyroge1480/.hermes/.env')\n"
        "if p.exists():\n"
        "  for line in p.read_text(errors='replace').splitlines():\n"
        "    line=line.strip()\n"
        "    if not line or line.startswith('#') or '=' not in line: continue\n"
        "    k,v=line.split('=',1)\n"
        "    os.environ.setdefault(k.strip(), v.strip().strip(\"'\\\"\"))\n"
        "os.environ.setdefault('HERMES_HOME','/home/jollyroge1480/.hermes')\n"
        "sys.path.insert(0,'/home/jollyroge1480/.hermes/hermes-agent')\n"
        "from tools.vision_tools import vision_analyze_tool\n"
        "print(asyncio.run(vision_analyze_tool(image_url=sys.argv[1], user_prompt=sys.argv[2])))\n"
    )
    quote = ""
    try:
        proc = subprocess.run(
            [_VISION_PY, "-c", script, path, prompt],
            capture_output=True, text=True, timeout=120,
        )
        raw = proc.stdout or ""
        start = raw.find("{")
        data = json.loads(raw[start:]) if start >= 0 else {}
        quote = _sanitize_vision(data.get("analysis") or "")
        if not quote:
            err = (proc.stderr or "")[-300:].replace("\n", " ")
            print(f"vision empty rc={proc.returncode} err={err}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"vision quote failed: {exc}", file=sys.stderr, flush=True)
    if not quote:
        quote = "NOT READ. Do not invent a part number or a product type."
    _VISION_CACHE[path] = quote
    return quote


def ground_photos(messages):
    """Replace the vague photo caption before the 9B can treat a guess as the part."""
    if not messages:
        return messages
    out = []
    for m in messages:
        m = dict(m)
        text = m.get("content")
        if m.get("role") != "user" or not isinstance(text, str) or "image_url:" not in text:
            out.append(m)
            continue
        paths = []
        for hit in _re.findall(r"image_url:\s*(\S+)", text):
            hit = hit.rstrip("]")
            if hit.startswith("/") and os.path.isfile(hit):
                paths.append(hit)
        quotes = [f"{os.path.basename(p)}: {_vision_quote(p)}" for p in paths]
        text = _re.sub(
            r"\[The user sent an image~ Here's what I can see:.*?\]\s*",
            "",
            text,
            flags=_re.S,
        )
        block = (
            "VISION QUOTE (only allowed source for a part number or product type):\n"
            + ("\n".join(quotes) if quotes else "NOT READ. Do not invent a part number or a product type.")
            + "\nIf the quote does not name the product, say the type is not readable. "
            "Do not guess switch, coil, or sensor.\n\n"
        )
        m["content"] = block + text
        print("vision grounded", file=sys.stderr, flush=True)
        out.append(m)
    return out


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
        # HIP_VISIBLE_DEVICES is passed through when set; unset means HIP's default device,
        # so a single-GPU box works without config (a hard "1" broke it).
        env={**os.environ, "QWYTHOS_MAXCTX": str(CTX)},
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
    out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
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


def trim_messages(messages, budget):
    """Keep tools plus the newest turns that fit. A long eBay topic otherwise
    prefills for minutes and holds the only decoder slot."""
    if not messages:
        return messages

    def ntok(msgs):
        return len(_tk.encode(chat_template(msgs)).ids)

    if ntok(messages) <= budget:
        return messages
    head = messages[:1] if messages[0].get("role") == "system" or "_rapier_tools" in messages[0] else []
    rest = messages[len(head):]
    if not rest:
        return messages
    lo, hi = 0, len(rest) - 1
    best = len(rest) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ntok(head + rest[mid:]) <= budget:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return head + rest[best:]


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(x) for x in content)
    return str(content or "")


def force_fit(messages, budget):
    """Fit the prompt in the 64k engine without throwing the tool list away.

    Hermes ships about 34k tokens of tools. This shape can hold that.
    Dropping the list is what made every tool turn a chat with no tools.
    Trim old turns. Leave the tool block.
    """
    if not messages:
        return messages
    fitted = [dict(m) for m in messages]
    cap = budget * 4
    for m in fitted:
        text = _as_text(m.get("content"))
        if len(text) > cap:
            m["content"] = text[-cap:]
    fitted = trim_messages(fitted, budget)
    guard = 0
    while guard < 16 and len(_tk.encode(chat_template(fitted)).ids) > budget:
        guard += 1
        idx = max(range(len(fitted)), key=lambda i: len(_as_text(fitted[i].get("content"))))
        text = _as_text(fitted[idx].get("content"))
        if len(fitted) > 1 and idx != len(fitted) - 1 and len(text) < 400:
            fitted.pop(idx)
            continue
        if len(text) < 200:
            break
        fitted[idx]["content"] = text[len(text) // 3:]
    if len(_tk.encode(chat_template(fitted)).ids) > budget:
        last = _as_text((fitted[-1].get("content") if fitted else "") or "hello")
        fitted = [{"role": "user", "content": last[-1500:]}]
    return fitted


def arm_deadline(proc, seconds):
    """Kill a run that stops emitting. One wedged generate holds the only slot."""
    def fire():
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def generate(messages, max_tokens, temperature, top_p, on_token=None):
    """Returns (text, raw_text) - raw includes any <tool_call> blocks.
    on_token(piece, in_think) streams decoded pieces as they generate (SSE path)."""
    global hist, primed
    max_tokens = min(max(int(max_tokens or 256), 1), GEN_CAP)
    kept = len(messages)
    messages = force_fit(messages, PROMPT_BUDGET)
    engine_start()
    ids = _tk.encode(chat_template(messages)).ids
    if len(ids) + max_tokens + 16 > CTX:
        raise RuntimeError(
            f"prompt is {len(ids)} tokens, this brain holds {CTX}. "
            "Start a new chat — the full catalog does not fit."
        )
    # find the longest shared prefix with the already-ingested history
    shared = 0
    for a, b in zip(hist, ids):
        if a != b:
            break
        shared += 1
    # Delta ingest is valid only when this prompt extends the engine's exact
    # context. A shared chat-template prefix is not that. Appending the new
    # question after the old answer made every reply the tail of "Pong".
    continuation = bool(hist) and shared == len(hist) and len(ids) > len(hist)
    need = (len(ids) - shared) if continuation else len(ids)
    if not continuation or len(hist) + need + max_tokens + 16 > CTX:
        engine_reset()
        new_tokens = ids
    else:
        new_tokens = ids[shared:]
    # This box is the MTP decoder. A sampled run is what pegs the GPU and
    # holds the only slot. eBay turns stay on the fast greedy path.
    greedy = True
    # First emitted id is the next token after the last ingested token — the
    # first token of this answer. Skipping it turned "Pong" into "ong".
    skip_first = False
    req = {"op": "run", "tokens": new_tokens, "gen": max_tokens + (1 if skip_first else 0),
           "mtp": bool(greedy)}
    # Live progress during the silent ingest: Hermes agent prompts are tens of thousands
    # of tokens and the bar otherwise sits at 0% until the first generated token.
    n_new = len(new_tokens)
    if on_token is not None and kept > len(messages):
        on_token(f"[rapier kept the newest turns, {len(ids)} prompt tokens]\n", True)
    elif on_token is not None and n_new > 256:
        eta = max(1, int(n_new / 140))
        on_token(f"[rapier ingesting {n_new} prompt tokens, ~{eta}s]\n", True)
    timer = arm_deadline(eng, DEADLINE)
    try:
        engine_send(req)
        out_ids, out_text, out_ids_raw = [], [], []
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
                out_ids.append(line["id"])   # history bookkeeping only; excluded from the reply text
                continue
            tid = line["id"]
            out_ids.append(tid)
            if not greedy and tid != EOS_ID and "top" in line and line["top"]:
                tid = sample(line["top"], temperature, top_p, rng)
            if tid == EOS_ID:
                eos_hit = True
                continue
            out_ids_raw.append(tid)
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
        # one batched decode for the whole reply (token-by-token decode was ~13 tok/s of host cost)
        full = _tk.decode(out_ids_raw) if out_ids_raw else ""
        return full, full
    except Exception:
        hist = []
        primed = False
        if eng is None or eng.poll() is not None:
            raise RuntimeError("rapier run killed after %.0fs with no finish" % DEADLINE)
        raise
    finally:
        timer.cancel()





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
        messages = ground_photos(body.get("messages", []))
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.6))
        top_p = float(body.get("top_p", 0.95))
        t0 = time.time()
        tools = slim_tools(body.get("tools") or None)
        if tools:
            messages = [dict(m) for m in messages]
            if messages:
                messages[0]["_rapier_tools"] = tools
            print(f"slim tools: kept {len(tools)}", file=sys.stderr, flush=True)
        want_stream = bool(body.get("stream"))
        if want_stream and os.environ.get("RAPIER_TRACE"):
            import sys as _s
            print("TRACE: stream request, generating...", file=_s.stderr, flush=True)
        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            write_lock = threading.Lock()
            def sse(obj):
                try:
                    with write_lock:
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
            stop_beat = threading.Event()
            def beat():
                while not stop_beat.wait(2):
                    on_tok(".", True)
            threading.Thread(target=beat, daemon=True).start()
            try:
                with eng_lock:
                    raw, _ = generate(messages, max_tokens, temperature, top_p, on_token=on_tok)
                if os.environ.get("RAPIER_TRACE"):
                    import sys as _s
                    print(f"TRACE: generate returned {len(raw)} chars", file=_s.stderr, flush=True)
            except Exception as e:
                # A raw {"error": ...} line makes the OpenAI client abort the
                # stream. Hermes then marks this brain dead and falls through
                # rate-limited cloud models. Deliver the failure as text.
                on_tok(str(e)[:500], False)
                sse({"id": "rapier", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
                return
            finally:
                stop_beat.set()
            calls, text = parse_tool_calls(raw) if tools else ([], raw)
            if "</think>" in text:
                _, _, text = text.partition("</think>")
                text = text.lstrip("\n")
            if text:
                on_tok(text, False)
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
        # non-streaming: fold the think block out of content like the streaming path does
        if "</think>" in text:
            head, _, tail = text.partition("</think>")
            text = (head.strip("\n") + "\n" if head.strip("\n") else "") + tail.lstrip("\n")
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
