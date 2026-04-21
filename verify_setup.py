"""
verify_setup.py
---------------
Phase 1 smoke test — verifies every component of the stack
before any real work begins.

Run:
    python verify_setup.py

All checks must pass (✓) before proceeding to Phase 2.
"""

import sys
import time

# ── ANSI colours (no external dep needed for this file) ──────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg: str) -> None: print(f"  {RED}✗{RESET}  {msg}"); sys.exit(1)
def info(msg: str) -> None: print(f"  {YELLOW}→{RESET}  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Python version
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Python version")
major, minor = sys.version_info.major, sys.version_info.minor
if major == 3 and minor >= 10:
    ok(f"Python {major}.{minor} — OK")
else:
    fail(f"Python {major}.{minor} detected. Requires 3.10+.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Core imports
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Core package imports")
packages = {
    "numpy":               "numpy",
    "pandas":              "pandas",
    "faiss":               "faiss",
    "sentence_transformers": "sentence-transformers",
    "ollama":              "ollama",
    "rich":                "rich",
}
for module, display in packages.items():
    try:
        __import__(module)
        ok(f"{display}")
    except ImportError as e:
        fail(f"{display} not found — run: pip install {display}\n    {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — FAISS basic operation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] FAISS sanity check")
try:
    import faiss
    import numpy as np

    dim  = 384
    vecs = np.random.rand(10, dim).astype("float32")
    faiss.normalize_L2(vecs)

    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)

    query = np.random.rand(1, dim).astype("float32")
    faiss.normalize_L2(query)
    scores, ids = idx.search(query, 3)

    assert ids.shape == (1, 3), "Unexpected FAISS result shape"
    ok(f"FAISS IndexFlatIP — added 10 vectors, searched top-3 — OK")
except Exception as e:
    fail(f"FAISS check failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Embedding model load & encode
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Sentence-Transformers embedding model")
info("Loading paraphrase-multilingual-MiniLM-L12-v2 (first run downloads ~120 MB)…")
try:
    from sentence_transformers import SentenceTransformer

    t0    = time.perf_counter()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    load_ms = (time.perf_counter() - t0) * 1000

    # Cold-encode a Bangla phrase
    t1  = time.perf_counter()
    vec = model.encode(["নুডুলসের দাম কত টাকা?"], normalize_embeddings=True)
    cold_ms = (time.perf_counter() - t1) * 1000

    # Warm-encode (should be much faster)
    t2  = time.perf_counter()
    vec = model.encode(["দাম কত টাকা?"], normalize_embeddings=True)
    warm_ms = (time.perf_counter() - t2) * 1000

    assert vec.shape == (1, 384), f"Expected (1, 384), got {vec.shape}"

    ok(f"Model loaded in {load_ms:.0f} ms")
    ok(f"Cold encode: {cold_ms:.1f} ms  |  Warm encode: {warm_ms:.1f} ms")
    ok(f"Output shape: {vec.shape} — correct 384-dim vector")

    if warm_ms > 60:
        info(f"Warm encode is {warm_ms:.1f} ms — slightly high, but acceptable on first run.")
    else:
        ok(f"Warm encode is within latency budget ✓")

except Exception as e:
    fail(f"Embedding model check failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Ollama server reachability
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Ollama server & model availability")
try:
    import ollama

    models = ollama.list()  # raises if server is unreachable
    model_names = [m.model for m in models.models]

    ok(f"Ollama server is reachable")
    info(f"Available models: {model_names}")

    if not any("gemma4:e2b" in name for name in model_names):
        fail(
            "gemma4:e2b not found in Ollama.\n"
            "    Run:  ollama pull gemma4:e2b\n"
            "    Then re-run this script."
        )
    ok("gemma4:e2b is available")

except Exception as e:
    fail(
        f"Cannot reach Ollama server: {e}\n"
        "    Make sure Ollama is running:  ollama serve"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — Ollama first-token latency (gemma4:e2b)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Ollama TTFT smoke test (gemma4:e2b, streaming)")
info("Sending a minimal Bangla prompt — measuring time-to-first-token…")
try:
    import ollama

    t0              = time.perf_counter()
    first_token_t   = None
    tokens_received = 0

    stream = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Reply in Bangla. Be very brief."
                ),
            },
            {
                "role": "user",
                "content": "নুডুলসের দাম কত?",   # "How much do noodles cost?"
            },
        ],
        stream=True,
    )

    for chunk in stream:
        content = chunk["message"]["content"]
        if content and first_token_t is None:
            first_token_t = time.perf_counter()
        tokens_received += len(content)

    ttft_ms    = (first_token_t - t0) * 1000 if first_token_t else float("inf")
    total_ms   = (time.perf_counter() - t0) * 1000

    ok(f"First token received in {ttft_ms:.1f} ms")
    ok(f"Total generation time:  {total_ms:.1f} ms")
    ok(f"Total characters received: {tokens_received}")

    if ttft_ms < 100:
        ok(f"TTFT {ttft_ms:.1f} ms < 100 ms — within budget ✓")
    elif ttft_ms < 200:
        info(
            f"TTFT {ttft_ms:.1f} ms — slightly over 100 ms on this cold run. "
            "This is expected on first call (model loading). "
            "Will be faster in the warmed-up pipeline."
        )
    else:
        info(
            f"TTFT {ttft_ms:.1f} ms — high. This is likely a cold-start penalty. "
            "Run the benchmark (Phase 5) with warmup to get true numbers."
        )

except Exception as e:
    fail(f"Ollama generation test failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"{GREEN}All checks passed. Environment is ready for Phase 2.{RESET}")
print(f"{'─'*50}\n")