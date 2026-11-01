from sentence_transformers import SentenceTransformer
import os
model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Preparing local embedding model: {model}")
SentenceTransformer(model)
print("Embedding model is ready.")
