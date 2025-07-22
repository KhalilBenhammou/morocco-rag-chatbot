from pathlib import Path
import json
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter


def load_raw_documents(data_dir="data/raw"):
    docs = []
    for path in Path(data_dir).iterdir():
        suffix = path.suffix.lower()
        if suffix == ".txt":
            loader = TextLoader(str(path))
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in {".html", ".htm"}:
            loader = UnstructuredHTMLLoader(str(path))
        else:
            continue
        docs.extend(loader.load())
    return docs


def split_docs(docs, chunk_size=1000, overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def save_chunks(chunks, out_path="data/chunks.jsonl"):
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"text": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    docs = load_raw_documents()
    chunks = split_docs(docs)
    save_chunks(chunks)
