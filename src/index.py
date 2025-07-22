# src/index.py
import json
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings


def load_chunks(path="data/chunks.jsonl"):
    texts, metas = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append(obj.get("metadata", {}))
    return texts, metas


if __name__ == "__main__":
    # 1. Load your chunked docs
    texts, metadatas = load_chunks()
    print(f"Loaded {len(texts)} chunks.")

    # 2. Initialize the embedding model wrapper
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = SentenceTransformerEmbeddings(model_name=model_name)
    print("Embedding model ready.")

    # 3. Build the FAISS index
    print("Building FAISS index…")
    index = FAISS.from_texts(texts, embedder, metadatas=metadatas)

    # 4. Save it locally
    index_path = "data/index/faiss_index"
    index.save_local(index_path)
    print(f"✅ FAISS index saved to {index_path}")
