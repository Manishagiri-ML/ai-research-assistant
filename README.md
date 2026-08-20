# 🤖 AI Research Assistant

An AI-powered research assistant that lets you upload a PDF, ask questions about it, and get instant summaries — powered by Retrieval-Augmented Generation (RAG).

## What it does

- **Upload any PDF** and the app extracts and indexes its content
- **Ask questions** about the document and get answers grounded in its actual content (not hallucinated)
- **Get AI-generated summaries** of the full document
- **Dynamic suggested questions** — after processing a document, the app uses an LLM to generate relevant questions specific to that document
- **Persistent history** — all uploaded papers and conversations are saved to a local database and survive app restarts

## How it works (RAG pipeline)

1. **Ingest**: extract text from the uploaded PDF and split it into overlapping chunks
2. **Embed**: convert each chunk into a vector using `sentence-transformers`
3. **Index**: store the vectors in a FAISS index for fast similarity search
4. **Retrieve**: given a question, find the most relevant chunks via semantic search
5. **Generate**: send the retrieved chunks + question to a free Hugging Face LLM, which answers using only that context

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web UI |
| pypdf | PDF text extraction |
| sentence-transformers | Free, local embeddings |
| FAISS | Vector similarity search |
| Hugging Face Inference API | Free hosted LLM for answering/summarizing |
| SQLite | Persistent storage for history and stats |

## Project structure

ai-research-assistant/
├── app.py # Streamlit UI
├── src/
│ ├── ingest.py # PDF loading + chunking
│ ├── embeddings.py # Embeddings + FAISS index
│ ├── retriever.py # Semantic search
│ ├── qa_chain.py # RAG-based Q&A
│ ├── summarizer.py # Summarization + suggested questions
│ └── database.py # SQLite persistence
├── data/raw/ # Uploaded PDFs
├── vectorstore/ # Saved FAISS index
└── requirements.txt


## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd ai-research-assistant

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token
# Create a .env file with:
# HUGGINGFACEHUB_API_TOKEN=your_token_here

# Run the app
streamlit run app.py
```

## Known limitations

- Summarization sends the full document text to the LLM at once, so very long documents may exceed the model's input limit
- Chunking is character-based rather than sentence-aware, which can occasionally split a sentence across two chunks
- Currently supports one document at a time

## Future improvements

- Sentence/paragraph-aware chunking
- Support for multiple documents at once
- Citation of which page/section an answer came from

---
Built by Manisha Giri