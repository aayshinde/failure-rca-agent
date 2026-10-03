"""Vector index over the maintenance documentation. Backends: FAISS (default) or Qdrant.

Usage:  python -m src.retrieval build            # embed docs -> index/
        python -m src.retrieval search "E203 fan fault"
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache

import numpy as np

from src import config
from src.docs_gen import chunk_docs


class LocalHashEmbeddings:
    """Offline TF-hashing embedder (no API). Fine for tests; use OpenAI embeddings for real runs."""

    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False, norm="l2",
                                     token_pattern=r"(?u)\b\w+\b")

    def embed_documents(self, texts):
        return self.vec.transform([t.replace("_", " ") for t in texts]).toarray().astype(np.float32)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder():
    if config.EMBED_MODEL == "local":
        return LocalHashEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    if config.EMBED_MODEL.startswith("ollama:"):   # e.g. ollama:nomic-embed-text  (free, semantic, local)
        return OpenAIEmbeddings(model=config.EMBED_MODEL[7:], base_url=config.OLLAMA_URL, api_key="ollama",
                                check_embedding_ctx_length=False)
    return OpenAIEmbeddings(model=config.EMBED_MODEL)


def index_dir():
    """One index per embedding model, so switching EMBED_MODEL never invalidates or overwrites another index."""
    d = config.INDEX_DIR / re.sub(r"[^A-Za-z0-9._-]+", "_", config.EMBED_MODEL)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unit(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class DocIndex:
    def __init__(self, chunks: list[dict], backend: str = config.VECTOR_BACKEND):
        self.chunks, self.backend = chunks, backend
        self._faiss = self._qdrant = None

    # ---------------------------------------------------------------- build / load
    @classmethod
    def build(cls, backend: str = config.VECTOR_BACKEND) -> "DocIndex":
        chunks = chunk_docs()
        vecs = _unit(get_embedder().embed_documents([c["text"] for c in chunks]))
        idx = cls(chunks, backend)
        (index_dir() / "chunks.json").write_text(json.dumps(chunks, indent=1))
        (index_dir() / "index_meta.json").write_text(json.dumps({"embed_model": config.EMBED_MODEL, "backend": backend,
                                                                     "n_chunks": len(chunks), "dim": int(vecs.shape[1])}))
        if backend == "faiss":
            import faiss

            idx._faiss = faiss.IndexFlatIP(vecs.shape[1])
            idx._faiss.add(vecs)
            faiss.write_index(idx._faiss, str(index_dir() / "docs.faiss"))
        else:
            from qdrant_client import models

            client = idx._qdrant_client()
            if client.collection_exists("docs"):
                client.delete_collection("docs")
            client.create_collection("docs", vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE))
            client.upsert("docs", points=[models.PointStruct(id=i, vector=v.tolist(), payload=c)
                                          for i, (v, c) in enumerate(zip(vecs, chunks))])
        return idx

    @classmethod
    def load(cls) -> "DocIndex":
        if not (index_dir() / "index_meta.json").exists():
            if config.EMBED_MODEL == "local" or config.EMBED_MODEL.startswith("ollama:"):
                return cls.build()         # free embedders: build on first use
            raise RuntimeError(f"No index for EMBED_MODEL={config.EMBED_MODEL}; run `python -m src.retrieval build`.")
        meta = json.loads((index_dir() / "index_meta.json").read_text())
        idx = cls(json.loads((index_dir() / "chunks.json").read_text()), meta["backend"])
        if idx.backend == "faiss":
            import faiss

            idx._faiss = faiss.read_index(str(index_dir() / "docs.faiss"))
        else:
            idx._qdrant_client()
        return idx

    def _qdrant_client(self):
        if self._qdrant is None:
            from qdrant_client import QdrantClient

            self._qdrant = QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL else \
                QdrantClient(path=str(index_dir() / "qdrant"))
        return self._qdrant

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = config.DOC_TOP_K) -> list[dict]:
        q = _unit(get_embedder().embed_query(query))
        if self.backend == "faiss":
            scores, ids = self._faiss.search(q[None, :], k)
            hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        else:
            res = self._qdrant_client().query_points("docs", query=q.tolist(), limit=k).points
            hits = [(int(p.id), float(p.score)) for p in res]
        return [{**self.chunks[i], "score": round(s, 4)} for i, s in hits]


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        idx = DocIndex.build()
        print(f"Indexed {len(idx.chunks)} chunks with {config.EMBED_MODEL} into {config.VECTOR_BACKEND}")
    else:
        for h in DocIndex.load().search(" ".join(sys.argv[2:])):
            print(f"{h['score']:.3f}  {h['chunk_id']}")


if __name__ == "__main__":
    main()
