import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ingest import load_pdf_text, chunk_text
from embeddings import create_embeddings, build_faiss_index, save_index
from qa_chain import answer_question
from summarizer import summarize_text, generate_suggested_questions
import database as db

db.init_db()

st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="wide")

# ---------- STYLING ----------
st.markdown("""
    <style>
    h1 {
        background: linear-gradient(90deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
    }
    .feature-card, .stat-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        height: 100%;
    }
    .feature-card h4 { margin-bottom: 0.3rem; color: var(--text-color); }
    .feature-card p { color: var(--text-color); opacity: 0.7; font-size: 0.9rem; }
    .stat-card .num { font-size: 1.8rem; font-weight: 800; color: #A78BFA; }
    .stat-card .label { font-weight: 600; }
    .stat-card .sub { opacity: 0.6; font-size: 0.8rem; }
    .stButton > button {
        background: linear-gradient(90deg, #7C3AED, #3B82F6);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: transform 0.15s ease;
    }
    .stButton > button:hover { transform: scale(1.03); color: white; }
    div[data-testid="stAlert"] { border-left: 4px solid #7C3AED; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "document_text" not in st.session_state:
    st.session_state["document_text"] = None
if "suggested_questions" not in st.session_state:
    st.session_state["suggested_questions"] = [
        "What is the main topic of this document?",
        "Summarize the key findings",
        "What methods are used?",
        "What are the limitations mentioned?",
    ]

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🤖 AI Research Assistant")
    st.caption("Your intelligent partner for research and learning.")
    st.divider()
    st.markdown("**Settings**")
    response_length = st.select_slider("Response Length", options=["Short", "Medium", "Long"], value="Medium")
    st.divider()
    st.info("💡 Tip: Upload a research paper (PDF) and ask any question about it!")
    st.divider()
    st.caption("⚡ Built by Manisha Giri · RAG · Hugging Face + FAISS + SQLite")

# ---------- HEADER ----------
st.title("🤖 AI Research Assistant")
st.markdown(
    "<p style='text-align:center; opacity:0.75;'>Upload papers, ask questions, get summaries, and accelerate your research 🚀</p>",
    unsafe_allow_html=True,
)

# ---------- STAT CARDS (REAL DATA) ----------
stats = db.get_stats()
s1, s2, s3, s4 = st.columns(4)
stat_cards = [
    (s1, stats["papers"], "Papers Analyzed", "Total uploaded papers"),
    (s2, stats["questions"], "Questions Asked", "Across all sessions"),
    (s3, stats["summaries"], "Summaries Generated", "From papers & text"),
    (s4, stats["active_days"], "Active Days", "Days you've used this app"),
]
for col, num, label, sub in stat_cards:
    with col:
        st.markdown(
            f"<div class='stat-card'><div class='num'>{num}</div>"
            f"<div class='label'>{label}</div><div class='sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------- FEATURE CARDS ----------
c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "📄", "Ask Anything", "Ask questions about your uploaded document and get grounded answers."),
    (c2, "☁️", "Upload & Analyze", "Upload a PDF and this app indexes it for instant search."),
    (c3, "🔍", "Semantic Search", "Finds the most relevant parts of your document by meaning, not keywords."),
    (c4, "🧠", "Summarize Text", "Condense long documents into a clear, short summary."),
]
for col, icon, title, desc in cards:
    with col:
        st.markdown(f"<div class='feature-card'><h4>{icon} {title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

st.divider()

# ---------- UPLOAD ----------
st.subheader("📤 Upload Your Document")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    save_path = os.path.join("data", "raw", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document", type="primary"):
        with st.spinner("Reading and indexing document..."):
            text = load_pdf_text(save_path)
            chunks = chunk_text(text)
            embeddings = create_embeddings(chunks)
            index = build_faiss_index(embeddings)
            save_index(index, chunks)
            st.session_state["document_text"] = text

            size_kb = round(uploaded_file.size / 1024, 1)
            db.log_paper(uploaded_file.name, size_kb)

        with st.spinner("Generating suggested questions..."):
            st.session_state["suggested_questions"] = generate_suggested_questions(text)

        st.success("Document processed! Ask a question or summarize below.")
        st.rerun()

st.divider()

# ---------- ASK SECTION ----------
st.subheader("💬 Ask the AI Research Assistant")

st.caption("Suggested questions:")
sc = st.columns(len(st.session_state["suggested_questions"]))
clicked_question = None
for i, sq in enumerate(st.session_state["suggested_questions"]):
    if sc[i].button(sq):
        clicked_question = sq

question = st.text_input("Type your research question here...", value=clicked_question or "")

col_ask, col_sum = st.columns(2)
with col_ask:
    if st.button("Ask", type="primary"):
        if question:
            with st.spinner("Thinking..."):
                answer = answer_question(question)
            st.info(answer)
            db.log_conversation(question, answer, "qa")
        else:
            st.warning("Please type or select a question first.")

with col_sum:
    if st.button("Summarize Document"):
        if st.session_state["document_text"]:
            with st.spinner("Summarizing..."):
                summary = summarize_text(st.session_state["document_text"])
            st.info(summary)
            db.log_conversation("(Summary requested)", summary, "summary")
        else:
            st.warning("Please upload and process a document first.")

st.divider()

# ---------- RECENTLY UPLOADED + RECENT CONVERSATIONS ----------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📄 Recently Uploaded Papers")
    papers = db.get_recent_papers()
    if papers:
        for filename, size_kb, uploaded_at in papers:
            st.markdown(f"**{filename}**  \n{size_kb} KB · {db.time_ago(uploaded_at)}")
    else:
        st.caption("No papers uploaded yet.")

with col_right:
    st.subheader("🕘 Recent Conversations")
    convos = db.get_recent_conversations()
    if convos:
        for q, a, created_at in convos:
            with st.expander(f"{q}  ·  {db.time_ago(created_at)}"):
                st.write(a)
    else:
        st.caption("No conversations yet.")