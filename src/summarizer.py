import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import InferenceClient

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(token=token)


def summarize_text(text):
    prompt = f"""Summarize the following document in 4-6 clear sentences.
Focus on the main ideas only, and write in plain language.

Document:
{text}

Summary:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )

    return response.choices[0].message.content


def generate_suggested_questions(text, num_questions=4):
    """
    Asks the LLM to read the document and suggest relevant questions
    someone might want to ask about it.
    """
    prompt = f"""Read the following document and suggest {num_questions} short, specific
questions someone might ask about it. Only return the questions, one per line,
numbered 1 to {num_questions}. Do not include any other text.

Document:
{text[:3000]}

Questions:"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    raw_output = response.choices[0].message.content

    questions = []
    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if line:
            cleaned = line.lstrip("0123456789.)- ").strip()
            if cleaned:
                questions.append(cleaned)

    return questions[:num_questions]


if __name__ == "__main__":
    from ingest import load_pdf_text

    text = load_pdf_text("data/raw/sample_document.pdf")
    summary = summarize_text(text)

    print("Summary:")
    print(summary)