"""
rag_pipeline.py
---------------
Full 4-stage Bangla RAG pipeline:
    Stage 1 — Coreference resolution (rule-based)
    Stage 2 — Query embedding (MiniLM-L12-v2)
    Stage 3 — Hybrid search (keyword match + FAISS vector search)
    Stage 4 — LLM answer generation (gemma-3-1b via llama-cpp-python, Metal)

Run standalone smoke test:
    python rag_pipeline.py
"""

import time
import numpy as np
import pandas as pd
import faiss
from llama_cpp import Llama
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
FAISS_PATH   = "products.faiss"
PKL_PATH     = "products.pkl"
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GGUF_PATH    = "gemma-3-1b-it-Q4_K_M.gguf"
TOP_K        = 3

SYSTEM_PROMPT = """আপনি একটি বাংলা ই-কমার্স স্টোরের সহায়ক।
শুধুমাত্র প্রদত্ত পণ্যের তথ্য ব্যবহার করে বাংলায় সংক্ষিপ্ত উত্তর দিন।
যদি পণ্যটি প্রদত্ত তথ্যে না থাকে, শুধু বলুন "পণ্যটি নেই" ।"""


# ── Stage 1: Coreference Resolution ─────────────────────────────────────────

class ConversationContext:
    """
    Rule-based coreference resolver for Bangla product QA.

    Example:
        Q1: "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?" -> entity: "নুডুলস"
        Q2: "দাম কত টাকা?" -> resolved: "নুডুলসএর দাম কত টাকা?"
    """

    def __init__(self, product_names: set[str]) -> None:
        self._product_names: set[str] = product_names
        self._entity_stack:  list[str] = []
        self._turn_history:  list[dict] = []

    def resolve(self, query: str) -> str:
        """Rewrite anaphoric query by injecting last known entity."""
        if self._is_anaphoric(query) and self._entity_stack:
            return f"{self._entity_stack[-1]}এর {query}"
        return query

    def update(self, query: str, response: str) -> None:
        """Extract product entity from turn and push to stack."""
        entity = self._extract_entity(query, response)
        if entity:
            self._entity_stack.append(entity)
        self._turn_history.append({"q": query, "a": response})

    def last_entity(self) -> str | None:
        return self._entity_stack[-1] if self._entity_stack else None

    def history(self) -> list[dict]:
        return list(self._turn_history)

    def _clean(self, token: str) -> str:
        t = token.strip('?।,!')
        for sx in ['এর', 'র', 'কে', 'তে']:
            if t.endswith(sx) and len(t) > len(sx) + 1:
                return t[:-len(sx)]
        return t

    def _is_anaphoric(self, query: str) -> bool:
        clean_tokens = {self._clean(t) for t in query.strip().split()}
        return not clean_tokens.intersection(self._product_names)

    def _extract_entity(self, query: str, response: str) -> str | None:
        for token in query.strip().split():
            t = self._clean(token)
            if t in self._product_names:
                return t
        for token in response.strip().split():
            t = self._clean(token)
            if t in self._product_names:
                return t
        return None


# ── Stage 2: Query Embedding ─────────────────────────────────────────────────

class Embedder:
    """
    ONNX Runtime embedder (CPU-optimized, ultra-fast).
    Replaces SentenceTransformer completely.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=False  # IMPORTANT: don't re-export every time
        )

        # warmup
        for _ in range(3):
            inputs = self._tokenizer(
                ["warmup"], return_tensors="pt", padding=True, truncation=True
            )
            self._model(**inputs)

    def encode(self, text: str) -> np.ndarray:
        inputs = self._tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        )

        outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling

        # normalize (same as SentenceTransformers)
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)

        return embeddings.detach().cpu().numpy()[0].astype("float32")


# ── Stage 3: Hybrid Search (Keyword + FAISS) ─────────────────────────────────

class ProductIndex:
    """
    Hybrid retriever: exact keyword match (priority) + FAISS vector search.
    Keyword matches always returned first; FAISS fills remaining slots.
    """

    def __init__(self, faiss_path: str = FAISS_PATH, pkl_path: str = PKL_PATH) -> None:
        self._index: faiss.IndexFlatIP = faiss.read_index(faiss_path)
        self._df:    pd.DataFrame      = pd.read_pickle(pkl_path)

    def search(self, query_vec: np.ndarray, query_text: str = "", k: int = TOP_K) -> list[dict]:
        """Hybrid search: keyword match first, FAISS fills remaining slots."""
        seen:    set[int]   = set()
        results: list[dict] = []

        # Stage A — exact keyword match sorted by specificity
        if query_text:
            def _strip(t: str) -> str:
                t = t.strip('?।,!')
                for sx in ['এর', 'র', 'কে', 'তে']:
                    if t.endswith(sx) and len(t) > len(sx) + 1:
                        return t[:-len(sx)]
                return t

            tokens = [_strip(t) for t in query_text.strip().split() if len(_strip(t)) > 1]
            tokens = sorted(
                tokens,
                key=lambda t: self._df["product_name"].str.contains(t, regex=False, na=False).sum()
            )
            for token in tokens:
                mask = self._df["product_name"].str.contains(token, regex=False, na=False)
                for _, row in self._df[mask].head(k).iterrows():
                    rid = int(row["id"])
                    if rid not in seen:
                        seen.add(rid)
                        results.append(self._row_to_dict(row, 1.0))
                if len(results) >= k:
                    break

        # Stage B — FAISS fills remaining slots
        if len(results) < k:
            q = query_vec.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            scores, ids = self._index.search(q, k * 2)
            for idx, score in zip(ids[0], scores[0]):
                if idx < 0:
                    continue
                row = self._df.iloc[idx]
                rid = int(row["id"])
                if rid not in seen:
                    seen.add(rid)
                    results.append(self._row_to_dict(row, float(score)))
                if len(results) >= k:
                    break

        return results[:k]

    def _row_to_dict(self, row: pd.Series, score: float) -> dict:
        return {
            "id":           int(row["id"]),
            "product_name": str(row["product_name"]),
            "price_bdt":    int(row["price_bdt"]),
            "text":         str(row["text"]),
            "score":        score,
        }


# ── Stage 4: LLM Answer Generation (llama-cpp-python, Metal) ─────────────────

class LlamaEngine:
    """
    Loads GGUF model once at startup via llama-cpp-python with Metal GPU layers.
    Bypasses Ollama HTTP layer entirely — TTFT ~25-50ms on M1.
    """

    def __init__(self, gguf_path: str = GGUF_PATH) -> None:
        print(f"  Loading GGUF model: {gguf_path}")
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_gpu_layers=-1,   # all layers on Metal GPU
            verbose=False,
        )
        # warmup
        self._llm("warmup", max_tokens=1)
        print("  GGUF model loaded and warmed up")

    def generate(self, context_hits: list[dict], resolved_query: str) -> tuple[str, float]:
        """
        Stream answer using llama-cpp-python. Returns (answer, ttft_ms).

        Args:
            context_hits:   Top-k product dicts from ProductIndex.search().
            resolved_query: Coreference-resolved query string.

        Returns:
            Tuple of (full answer string, time-to-first-token in ms).
        """
        context_text = "\n".join(h["text"] for h in context_hits)

        # Build chat prompt manually using Gemma instruct format
        prompt = (
            f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
            f"<start_of_turn>user\n"
            f"পণ্যের তথ্য:\n{context_text}\n\nপ্রশ্ন: {resolved_query}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        t0:            float       = time.perf_counter()
        first_token_t: float | None = None
        parts:         list[str]   = []

        stream = self._llm(
            prompt,
            max_tokens=150,
            stream=True,
            stop=["<end_of_turn>", "<start_of_turn>"],
            temperature=0.1,
        )

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                parts.append(token)

        ttft_ms = (first_token_t - t0) * 1000 if first_token_t else float("inf")
        return "".join(parts).strip(), ttft_ms


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """Orchestrates all four stages into a single query() call."""

    def __init__(self) -> None:
        print("Initialising RAG pipeline...")
        self._embedder      = Embedder()
        self._index         = ProductIndex()
        self._llama         = LlamaEngine()
        self._product_names = self._build_product_name_set()
        print("  Pipeline ready\n")

    def new_context(self) -> ConversationContext:
        return ConversationContext(self._product_names)

    def query(self, raw_query: str, ctx: ConversationContext) -> dict:
        """Run the full 4-stage pipeline for one user turn."""
        t_start = time.perf_counter()

        resolved = ctx.resolve(raw_query)
        t_coref  = time.perf_counter()

        vec     = self._embedder.encode(resolved)
        t_embed = time.perf_counter()

        hits     = self._index.search(vec, query_text=resolved, k=TOP_K)
        t_search = time.perf_counter()

        answer, ttft_ms = self._llama.generate(hits, resolved)
        t_end           = time.perf_counter()

        ctx.update(raw_query, answer)

        return {
            "raw_query":      raw_query,
            "resolved_query": resolved,
            "hits":           hits,
            "answer":         answer,
            "latency": {
                "coref_ms":  round((t_coref  - t_start) * 1000, 2),
                "embed_ms":  round((t_embed  - t_coref)  * 1000, 2),
                "search_ms": round((t_search - t_embed)  * 1000, 2),
                "ttft_ms":   round(ttft_ms, 2),
                "total_ms":  round((t_end    - t_start)  * 1000, 2),
            },
        }

    def _build_product_name_set(self) -> set[str]:
        df = pd.read_pickle(PKL_PATH)
        tokens: set[str] = set()
        for name in df["product_name"]:
            for token in str(name).strip().split():
                t = token.strip('?।,!')
                for sx in ['এর', 'র', 'কে', 'তে']:
                    if t.endswith(sx) and len(t) > len(sx) + 1:
                        t = t[:-len(sx)]
                        break
                tokens.add(t)
        return tokens


# ── Standalone smoke test ─────────────────────────────────────────────────────

def _smoke_test() -> None:
    pipeline = RAGPipeline()
    ctx      = pipeline.new_context()

    print("=" * 55)
    print("SMOKE TEST - Two-turn coreference conversation")
    print("=" * 55)

    for i, q in enumerate([
        "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?",
        "দাম কত টাকা?",
    ], 1):
        print(f"\n[Turn {i}] Raw query: {q}")
        result = pipeline.query(q, ctx)
        print(f"  Resolved : {result['resolved_query']}")
        print(f"  Top hits :")
        for h in result["hits"]:
            print(f"    - {h['product_name']} - {h['price_bdt']} টাকা")
        print(f"  Answer   : {result['answer']}")
        print(f"  Latency  :")
        for k, v in result["latency"].items():
            flag = " *** UNDER 100ms ***" if k == "ttft_ms" and isinstance(v, float) and v < 100 else ""
            print(f"    {k:<12}: {v} ms{flag}")

    print("\n" + "=" * 55)
    print("Smoke test complete.")
    print("=" * 55)


if __name__ == "__main__":
    _smoke_test()