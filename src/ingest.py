from pypdf import PdfReader

def load_pdf_text(file_path):
    """
    Opens a PDF file and returns all its text as one big string.
    """
    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        full_text += page_text + "\n"

    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into smaller pieces (chunks) so it's easier to search later.
    chunk_size = how many characters per chunk
    overlap = how many characters repeat between chunks, so we don't cut off
              a sentence's meaning right at the boundary
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move forward, but re-include a little overlap

    return chunks

# This block only runs when we run THIS file directly (for testing)
if __name__ == "__main__":
    text = load_pdf_text("data/raw/sample_document.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")
    print("\n--- First chunk ---")
    print(chunks[0])
    print("\n--- Second chunk ---")
    print(chunks[1])