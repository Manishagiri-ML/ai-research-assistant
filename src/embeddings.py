from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def save_index(index, chunks, index_path="vectorstore/index.faiss", chunks_path="vectorstore/chunks.pkl"):
    """
    Saves the FAISS index AND the original text chunks to disk.
    We need both: the index finds WHICH chunk matches, but only the
    saved chunks list lets us get back the actual TEXT of that chunk.
    """
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)


def load_index(index_path="vectorstore/index.faiss", chunks_path="vectorstore/chunks.pkl"):
    """
    Loads a previously saved index and its chunks back from disk.
    """
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    from ingest import load_pdf_text, chunk_text

    text = load_pdf_text("data/raw/sample_document.pdf")
    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    save_index(index, chunks)
    print("Index and chunks saved to vectorstore/")