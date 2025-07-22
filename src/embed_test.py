# src/embed_test.py
import json
from sentence_transformers import SentenceTransformer


def load_chunks(path="data/chunks.jsonl", n=5):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(json.loads(line)["text"])
    return lines


if __name__ == "__main__":
    # 1. Load a few chunk texts
    texts = load_chunks()
    print(f"Loaded {len(texts)} chunks for testing.")

    # 2. Load the embedding model
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading model {model_name}…")
    model = SentenceTransformer(model_name)

    # 3. Encode the chunks
    embeddings = model.encode(texts)
    print("Embeddings shape:", embeddings.shape)
    # Optionally print the first vector
    print("First embedding (truncated):", embeddings[0][:5])
