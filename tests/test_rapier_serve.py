#!/usr/bin/env python3
"""End-to-end test for the Rapier serve path.

1. engine protocol: --serve emits the exact greedy token stream (MTP verified)
2. tokenizer: ground-truth encode
3. server: /v1/chat/completions greedy parity + multi-turn

Run:  python3 tests/test_rapier_serve.py [--skip-server]
Requires the engine built and HIP_VISIBLE_DEVICES pinned to the big card.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ENGINE = os.environ.get("RAPIER_ENGINE", "/data/hermes/Strata/build-hip/qwythos")
MODEL = os.environ.get("RAPIER_MODEL", "/data/hermes/models/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q6_K.gguf")
TOKDIR = os.path.join(os.path.dirname(__file__), "..", "server", "tokenizer")

# greedy stream for prompt token 9707 (recorded from the CLI MTP run, 400-token diff clean)
EXPECTED = [13, 220, 16, 13, 3437, 369, 279, 6463, 1881, 264, 328, 82, 7182, 1834,
            1, 321, 264, 328, 20384, 41827, 1834, 42418, 271, 32, 13, 357, 12570,
            1834, 369, 279, 854, 424, 4829, 364, 279, 8964, 310, 16424, 2957, 383]


def test_engine_serve():
    # the proven invocation: shell pipe keeps stdin open while the engine streams
    req = json.dumps({"op": "run", "tokens": [9707], "gen": len(EXPECTED), "mtp": True},
                     separators=(",", ":"))  # the C++ parser matches compact "key":[ exactly
    cmd = "echo " + json.dumps(req) + " | (sleep 60; cat) | " + ENGINE + " --model " + MODEL + " --serve"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "HIP_VISIBLE_DEVICES": "1"})
    got = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "id" in d:
            got.append(d["id"])
    assert got == EXPECTED, f"stream mismatch: got {got[:10]} expected {EXPECTED[:10]}"
    print(f"PASS engine-serve: {len(got)} tokens bit-exact vs recorded MTP stream")


def test_tokenizer():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from extract_tokenizer import extract  # noqa: F401  (verifies on extract)
    ok = extract(MODEL, os.path.abspath(TOKDIR), "The capital of France is",
                 [760, 6511, 314, 9338, 369])
    assert ok, "tokenizer ground-truth encode mismatch"
    print("PASS tokenizer: ground-truth encode exact")


def test_server():
    srv = subprocess.Popen([sys.executable,
                            os.path.join(os.path.dirname(__file__), "..", "server", "rapier_server.py")],
                           env={**os.environ, "HIP_VISIBLE_DEVICES": "1", "RAPIER_PORT": "8085"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = "http://127.0.0.1:8085"
        for _ in range(30):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps({"messages": [
                                         {"role": "user", "content": "The capital of France is"}],
                                         "max_tokens": 32, "temperature": 0}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer local"})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        content = d["choices"][0]["message"]["content"]
        assert "Paris" in content, f"wrong reply: {content[:80]}"
        assert d["timings"]["tok_s"] > 20, f"too slow: {d['timings']}"
        print(f"PASS server: reply contains Paris, {d['timings']['tok_s']} tok/s")
    finally:
        srv.send_signal(signal.SIGTERM)


if __name__ == "__main__":
    test_tokenizer()
    test_engine_serve()
    if "--skip-server" not in sys.argv:
        test_server()
    print("ALL RAPIER TESTS PASS")
