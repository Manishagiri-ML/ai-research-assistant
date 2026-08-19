from embeddings import model, load_index
import numpy as np


def search(query, top_k=3):
    """
    Given a question (query), finds the most relevant chunks from our
    saved document index.
    top_k = how many matching chunks to return (3 is a good starting point)
    """
    index, chunks = load_index()

    # Turn the QUESTION into a vector the same way we did for the document chunks
    query_vector = model.encode([query])

    # Ask FAISS: which stored vectors are closest to this one?
    distances, indices = index.search(np.array(query_vector), top_k)

    # indices[0] holds the positions of the best-matching chunks
    results = [chunks[i] for i in indices[0]]

    return results


if __name__ == "__main__":
    question = "What is RAG and why does it matter?"
    results = search(question)

    print(f"Question: {question}\n")
    for i, chunk in enumerate(results):
        print(f"--- Match {i+1} ---")
        print(chunk)
        print()