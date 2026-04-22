"""
fix_index.py
------------
Rebuilds products.faiss using a combined embedding field:
    "{product_name} — {text}"
This ensures product name tokens (e.g. "নুডুলস") dominate the
vector representation and retrieval is accurate.

Run:
    python fix_index.py
"""

import time
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

CSV_PATH   = "products.csv"
FAISS_PATH = "products.faiss"
PKL_PATH   = "products.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64

def main() -> None:
    print("Loading dataset…")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"  ✓ {len(df):,} records loaded")

    # Combine product_name + text so the name is heavily weighted
    df["embed_text"] = df["product_name"] + " — " + df["text"]

    print(f"\nLoading model: {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    model.encode(["warmup"], normalize_embeddings=True)
    print("  ✓ Model warmed up")

    print(f"\nEmbedding {len(df):,} records…")
    t0   = time.perf_counter()
    vecs = model.encode(
        df["embed_text"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    print(f"  ✓ Done in {time.perf_counter()-t0:.1f}s | shape: {vecs.shape}")

    print("\nBuilding FAISS IndexFlatIP…")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    print(f"  ✓ {index.ntotal:,} vectors indexed")

    # Quick verification
    print("\nVerification queries:")
    for query in ["নুডুলস", "ব্লুটুথ ইয়ারফোন", "মশার কয়েল"]:
        vec = model.encode([query], normalize_embeddings=True,
                           convert_to_numpy=True).astype("float32")
        scores, ids = index.search(vec, 3)
        print(f"\n  Query: {query}")
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
            row = df.iloc[idx]
            print(f"    [{rank}] {row['product_name']} — {row['price_bdt']} টাকা (score: {score:.4f})")

    # Persist
    faiss.write_index(index, FAISS_PATH)
    df.to_pickle(PKL_PATH)
    print(f"\n✓ Saved {FAISS_PATH} and {PKL_PATH}")

if __name__ == "__main__":
    main()