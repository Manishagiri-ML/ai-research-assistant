import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from retriever import search

# Load the .env file so we can read our token
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=token)


def answer_question(question, top_k=3):
    context_chunks = search(question, top_k=top_k)
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )

    answer = response.choices[0].message.content

    # Return both the answer AND the source chunks it was grounded in
    return {
        "answer": answer,
        "sources": context_chunks,
    }


if __name__ == "__main__":
    question = "What is RAG and why does it matter?"
    answer = answer_question(question)

    print(f"Question: {question}\n")
    print(f"Answer: {answer}")