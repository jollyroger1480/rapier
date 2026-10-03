#!/usr/bin/env python3
"""Extract the Qwen BPE tokenizer from a qwen35/qwen4exp GGUF into a HuggingFace
tokenizers-compatible tokenizer.json (ByteLevel BPE, Qwen pretokenizer).

Manual GGUF metadata parse (string arrays), no gguf-py API dependence.

Verify: "The capital of France is" -> [760, 6511, 314, 9338, 369] (Qwythos-9B-v2).
"""
import json, struct, sys, os

def read_meta_strings(gguf_path):
    with open(gguf_path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', 'not a GGUF'
        ver, = struct.unpack('<I', f.read(4))
        n_tensors, = struct.unpack('<Q', f.read(8))
        n_kv, = struct.unpack('<Q', f.read(8))
        def rstr():
            n, = struct.unpack('<Q', f.read(8))
            return f.read(n).decode('utf-8', 'replace')
        def skip_val(t):
            if t == 8: rstr(); return
            if t == 9:  # array
                et, = struct.unpack('<I', f.read(4))
                n, = struct.unpack('<Q', f.read(8))
                if et == 8:
                    for _ in range(n): rstr()
                elif et == 9:
                    for _ in range(n): skip_val(et)
                else:
                    widths = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    f.read(n * widths[et])
                return
            widths = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
            f.read(widths[t])
        out = {}
        for _ in range(n_kv):
            name = rstr()
            t, = struct.unpack('<I', f.read(4))
            if t == 8:
                out[name] = ('str', rstr())
            elif t == 9:
                et, = struct.unpack('<I', f.read(4))
                n, = struct.unpack('<Q', f.read(8))
                if et == 8:
                    vals = [rstr() for _ in range(n)]
                elif et == 9:
                    vals = None
                    for _ in range(n): skip_val(et)
                else:
                    widths = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    fmt = {0:'b',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[et]
                    vals = list(struct.unpack('<' + fmt * n, f.read(n * widths[et]))) if n else []
                out[name] = ('arr', vals)
            else:
                widths = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                fmt = {0:'b',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
                v, = struct.unpack('<' + fmt, f.read(widths[t]))
                out[name] = ('val', v)
        return out

def extract(gguf_path, out_dir, ref_text=None, ref_ids=None):
    meta = read_meta_strings(gguf_path)
    tokens = meta['tokenizer.ggml.tokens'][1]
    merges = meta['tokenizer.ggml.merges'][1]
    tt = meta.get('tokenizer.ggml.token_type')
    tv = tt[1] if tt else []
    assert tokens and merges, 'GGUF lacks tokenizer strings'
    vocab = {t: i for i, t in enumerate(tokens)}
    special = {}
    for i, ttype in enumerate(tv[:len(tokens)]):
        if ttype in (2, 3):
            special[tokens[i]] = i
    for name in ('<|im_end|>', '<|im_start|>'):
        if name in vocab:
            special.setdefault(name, vocab[name])
    tj = {
        "version": "1.0", "truncation": None,
        "added_tokens": [{"id": i, "special": True, "content": c, "single_word": False,
                          "lstrip": False, "rstrip": False, "normalized": False}
                         for c, i in sorted(special.items(), key=lambda kv: kv[1])],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
             "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False,
                  "vocab": vocab, "merges": merges},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'tokenizer.json'), 'w') as f:
        json.dump(tj, f, ensure_ascii=False)
    print(f"tokens {len(tokens)} merges {len(merges)} special {len(special)}")
    if ref_text and ref_ids:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(out_dir, 'tokenizer.json'))
        got = tk.encode(ref_text).ids
        ok = got == ref_ids
        print(f"verify: {got} vs {ref_ids} -> {'PASS' if ok else 'FAIL'}")
        return ok
    return True

if __name__ == '__main__':
    gg = sys.argv[1] if len(sys.argv) > 1 else '/data/hermes/models/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q6_K.gguf'
    out = sys.argv[2] if len(sys.argv) > 2 else '/data/hermes/rapier/server/tokenizer'
    ok = extract(gg, out, 'The capital of France is', [760, 6511, 314, 9338, 369])
    sys.exit(0 if ok else 1)
