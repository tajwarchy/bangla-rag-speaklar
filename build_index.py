"""
build_index.py
--------------
Embeds all 5,000 product records from products.csv using
paraphrase-multilingual-MiniLM-L12-v2 and builds a FAISS
IndexFlatIP held entirely in RAM, then persists both:
    products.faiss  — FAISS index (searchable vectors)
    products.pkl    — DataFrame (for result lookup)

Run:
    python build_index.py

Prerequisites:
    products.csv must exist (run dataset_gen.py first)
"""

import time
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH     = "products.csv"
FAISS_PATH   = "products.faiss"
PKL_PATH     = "products.pkl"
MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE   = 64
EMBED_DIM    = 384


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the product CSV."""
    print(f"Loading dataset from {path}…")
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"id", "product_name", "price_bdt", "text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    print(f"  ✓ Loaded {len(df):,} records | columns: {list(df.columns)}")
    return df


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised float32 vectors.
    Returns shape (N, EMBED_DIM).
    """
    print(f"\nEmbedding {len(texts):,} texts (batch_size={BATCH_SIZE})…")
    t0   = time.perf_counter()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    vecs    = vecs.astype("float32")

    print(f"  ✓ Embedded in {elapsed:.1f}s | shape: {vecs.shape}")
    print(f"  ✓ Memory footprint: {vecs.nbytes / 1024 / 1024:.2f} MB")
    return vecs


def build_faiss_index(vecs: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product FAISS index.
    Vectors must already be L2-normalised (inner product == cosine similarity).
    """
    print("\nBuilding FAISS IndexFlatIP…")
    t0    = time.perf_counter()
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    elapsed = time.perf_counter() - t0

    print(f"  ✓ Index built in {elapsed*1000:.1f} ms")
    print(f"  ✓ Total vectors in index: {index.ntotal:,}")
    return index


def verify_index(
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    df: pd.DataFrame,
) -> None:
    """Smoke-test the index with a known Bangla query."""
    print("\nVerifying index with test query…")

    test_queries = [
        "নুডুলসের দাম কত?",
        "ব্লুটুথ ইয়ারফোন কিনতে চাই",
        "মশার কয়েলের দাম কত টাকা?",
    ]

    for query in test_queries:
        vec = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        t0 = time.perf_counter()
        scores, ids = index.search(vec, 3)
        search_ms   = (time.perf_counter() - t0) * 1000

        print(f"\n  Query : {query}")
        print(f"  Search: {search_ms:.2f} ms")
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
            row = df.iloc[idx]
            print(f"    [{rank}] {row['product_name']} — {row['price_bdt']} টাকা  (score: {score:.4f})")


def persist(index: faiss.IndexFlatIP, df: pd.DataFrame) -> None:
    """Write FAISS index and DataFrame to disk."""
    print(f"\nPersisting index → {FAISS_PATH}")
    faiss.write_index(index, FAISS_PATH)
    print(f"  ✓ Saved {FAISS_PATH}")

    print(f"Persisting DataFrame → {PKL_PATH}")
    df.to_pickle(PKL_PATH)
    print(f"  ✓ Saved {PKL_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.perf_counter()

    # 1. Load dataset
    df = load_dataset(CSV_PATH)

    # 2. Load embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}…")
    t0    = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    # 3. Warmup
    print("\nWarming up model…")
    model.encode(["warmup"], normalize_embeddings=True)
    print("  ✓ Warmup complete")

    # 4. Embed all product texts
    vecs = embed_texts(model, df["text"].tolist())

    # 5. Build FAISS index
    index = build_faiss_index(vecs)

    # 6. Verify with test queries
    verify_index(index, model, df)

    # 7. Persist to disk
    persist(index, df)

    total_s = time.perf_counter() - t_total
    print(f"\n{'─'*50}")
    print(f"✓ Index build complete in {total_s:.1f}s")
    print(f"  products.faiss  — {index.ntotal:,} vectors @ {EMBED_DIM}-dim")
    print(f"  products.pkl    — {len(df):,} product records")
    print(f"{'─'*50}")
    print("\nReady for Phase 4 — RAG Pipeline.\n")


if __name__ == "__main__":
    main()