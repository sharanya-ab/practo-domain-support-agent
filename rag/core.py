from __future__ import annotations
import os, re, json, hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / ".chroma")))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FIXED_COLLECTION = "practo_fixed_overlap"
SENTENCE_COLLECTION = "practo_sentence_chunks"

@lru_cache(maxsize=1)
def embedder():
    return SentenceTransformer(MODEL_NAME)

def load_documents() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8").strip()
            for p in sorted(KB_DIR.glob("*.txt"))}

def fixed_chunks(text: str, size: int = 420, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        part = words[start:start+size]
        if part:
            chunks.append(" ".join(part))
        if start + size >= len(words): break
        start += size - overlap
    return chunks

def sentence_chunks(text: str, max_words: int = 65) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out, buf, count = [], [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if buf and count + n > max_words:
            out.append(" ".join(buf)); buf, count = [], 0
        buf.append(sentence); count += n
    if buf: out.append(" ".join(buf))
    return out

def _client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def build_indexes():
    client = _client()
    docs = load_documents()
    summary = {}
    for collection_name, chunker in [
        (FIXED_COLLECTION, fixed_chunks), (SENTENCE_COLLECTION, sentence_chunks)
    ]:
        try: client.delete_collection(collection_name)
        except Exception: pass
        col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        ids, texts, metas = [], [], []
        for doc_id, content in docs.items():
            for idx, chunk in enumerate(chunker(content)):
                ids.append(hashlib.md5(f"{collection_name}:{doc_id}:{idx}".encode()).hexdigest())
                texts.append(chunk)
                metas.append({"document_id": doc_id, "chunk_index": idx})
        vectors = embedder().encode(texts, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)
        summary[collection_name] = {"documents": len(docs), "chunks": len(ids)}
    return summary

def get_collection(strategy="fixed_overlap"):
    name = FIXED_COLLECTION if strategy == "fixed_overlap" else SENTENCE_COLLECTION
    return _client().get_collection(name)

def retrieve(query: str, strategy="fixed_overlap", top_k=4):
    col = get_collection(strategy)
    vector = embedder().encode([query], normalize_embeddings=True).tolist()[0]
    result = col.query(query_embeddings=[vector], n_results=min(top_k, col.count()),
                       include=["documents", "metadatas", "distances"])
    rows = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        rows.append({"text": doc, "document_id": meta["document_id"],
                     "distance": float(dist), "similarity": float(1.0-dist)})
    return rows

def calibrate_threshold():
    in_scope = [
        "How far in advance should I cancel or reschedule an appointment?",
        "What are the consultation fees for Cardiology?",
        "How do I submit an insurance claim?",
        "How long do routine laboratory results take?",
        "Can I request a second opinion?"
    ]
    out_scope = [
        "Who won the cricket world cup and what was the final score?",
        "Write a Python program to sort a list of integers."
    ]
    ins = [retrieve(q, top_k=1)[0]["similarity"] for q in in_scope]
    outs = [retrieve(q, top_k=1)[0]["similarity"] for q in out_scope]
    # Similarity distributions must be separated to claim a measured threshold.
    low_in, high_out = min(ins), max(outs)
    if low_in <= high_out:
        raise RuntimeError(
            f"Similarity clusters overlap: min(in-scope)={low_in:.4f}, "
            f"max(out-of-scope)={high_out:.4f}. Revise KB/probes/model; do not guess."
        )
    threshold = (low_in + high_out) / 2
    report = {"in_scope": dict(zip(in_scope, ins)), "out_of_scope": dict(zip(out_scope, outs)),
              "min_in_scope": low_in, "max_out_of_scope": high_out,
              "chosen_threshold": threshold}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/rag_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return threshold, report

def grounded_answer(query: str, strategy="fixed_overlap", threshold=None, top_k=4):
    if threshold is None:
        p = Path("artifacts/rag_calibration.json")
        if not p.exists(): raise RuntimeError("Run build_index.py first to calibrate threshold.")
        threshold = json.loads(p.read_text(encoding="utf-8"))["chosen_threshold"]
    rows = retrieve(query, strategy, top_k)
    if not rows or rows[0]["similarity"] < threshold:
        return {"answer": "I don't know based on the available clinic policy documents.",
                "grounded": False, "retrieved": rows}
    # Deterministic extractive mock generation: every factual sentence is a sentence
    # from retrieved text; no unsupported claims are added.
    
    # Keep evidence from the highest-ranked policy document.
    # This avoids mixing unrelated policies into a single answer.
    primary_document = rows[0]["document_id"]

    relevant_rows = [
        row for row in rows
        if row["document_id"] == primary_document
        and row["similarity"] >= threshold
    ]

    answer = " ".join(row["text"] for row in relevant_rows)

    return {
        "answer": answer,
        "grounded": True,
        "retrieved": relevant_rows,
    }


def doc_level_metrics(retrieved, relevant_docs, k=4):
    got = list(dict.fromkeys(r["document_id"] for r in retrieved[:k]))
    relevant = set(relevant_docs)
    tp = len(set(got) & relevant)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    return {"retrieved_documents": got, "relevant_documents": sorted(relevant),
            "arithmetic": f"precision={tp}/{len(got) if got else 0}; recall={tp}/{len(relevant) if relevant else 0}",
            "precision": precision, "recall": recall}
