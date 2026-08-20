import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingest import load_pdf_text, chunk_text
from embeddings import create_embeddings, build_faiss_index, save_index
from qa_chain import answer_question
from summarizer import summarize_text, generate_suggested_questions
from retriever import search
import database as db

db.init_db()
db.init_notes_table()

st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="wide")

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
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# ---------- SIDEBAR NAV ----------
with st.sidebar:
    st.markdown("### 🤖 AI Research Assistant")
    st.caption("Your intelligent partner for research and learning.")
    st.divider()
    st.markdown("**NAVIGATION**")

    pages = ["Home", "Ask Assistant", "Upload & Analyze", "Search Papers", "Summarize Text", "Saved Notes", "History"]
    for p in pages:
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        if st.button(p, key=f"nav_{p}", type=("primary" if st.session_state["page"] == p else "secondary")):
            st.session_state["page"] = p
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**SETTINGS**")
    response_length = st.select_slider("Response Length", options=["Short", "Medium", "Long"], value="Medium")
    st.divider()
    st.info("💡 Tip: Upload a research paper (PDF) and ask any question about it!")
    st.divider()
    st.caption("⚡ Built by Manisha Giri · RAG · Hugging Face + FAISS + SQLite")

# ---------- COLORS BASED ON THEME ----------
if st.session_state["theme"] == "light":
    bg_color = "#FFFFFF"
    card_bg = "#FFFFFF"
    text_color = "#111827"
    border_color = "rgba(124, 58, 237, 0.3)"
else:
    bg_color = "#000000"
    card_bg = "#000000"
    text_color = "#E5E7EB"
    border_color = "rgba(124, 58, 237, 0.25)"

# ---------- STYLING ----------
st.markdown(f"""
    <style>
    .block-container {{
        padding-top: 2rem !important;
    }}
       .stApp, [data-testid="stAppViewContainer"], .main {{
        background: linear-gradient(-45deg, {bg_color}, {card_bg}, {bg_color}, {card_bg}) !important;
        background-size: 400% 400% !important;
        animation: gradientShift 18s ease infinite !important;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    h1, h2 {{
        background: linear-gradient(90deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}
    p, span, label, .stMarkdown {{
        color: {text_color} !important;
    }}
    .feature-card, .stat-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 1.2rem;
        height: 100%;
    }}
    .feature-card h4 {{ margin-bottom: 0.3rem; color: {text_color}; }}
    .feature-card p {{ color: {text_color}; opacity: 0.7; font-size: 0.9rem; }}
    .stat-card .num {{ font-size: 1.8rem; font-weight: 800; color: #A78BFA; }}
    .stat-card .label {{ font-weight: 600; color: {text_color}; }}
    .stat-card .sub {{ opacity: 0.6; font-size: 0.8rem; color: {text_color}; }}
    section[data-testid="stSidebar"] {{
        background: {card_bg};
    }}
    .stButton > button {{
        background: linear-gradient(90deg, #7C3AED, #3B82F6);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: transform 0.15s ease;
    }}
    .stButton > button:hover {{ transform: scale(1.03); color: white; }}
    div[data-testid="stAlert"] {{ border-left: 4px solid #7C3AED; border-radius: 6px; }}
    .nav-btn button {{
        width: 100%;
        text-align: left !important;
        background: transparent !important;
        color: {text_color} !important;
        border: 1px solid transparent !important;
        font-weight: 500 !important;
    }}
    .nav-btn button:hover {{
        background: rgba(124, 58, 237, 0.15) !important;
    }}
        header[data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
      .theme-btn button {{
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        max-height: 42px !important;
        padding: 0 !important;
        margin: 0 !important;
        line-height: 42px !important;
        font-size: 1.1rem !important;
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    .theme-btn button p {{
        margin: 0 !important;
        line-height: 1 !important;
    }}
    .theme-btn button:hover {{
        transform: scale(1.08);
    }}
        [data-testid="stFileUploaderDropzone"] {{
        background-color: {card_bg} !important;
        border: 1px dashed {border_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {text_color} !important;
    }}
        [data-testid="stFileUploaderDropzone"] button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
        background: linear-gradient(90deg, #7C3AED, #3B82F6) !important;
        color: white !important;
        transform: scale(1.03);
        transition: all 0.15s ease;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- THEME TOGGLE BUTTON (top-right of content) ----------
top_l, top_r = st.columns([15, 1])
with top_r:
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    icon = "🌙" if st.session_state["theme"] == "dark" else "☀️"
    if st.button(icon, key="theme_icon_btn"):
        st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state["page"]

# ---------- HOME ----------
if page == "Home":
    st.title("🤖 AI Research Assistant")
    st.markdown(
        "<p style='opacity:0.75;'>Upload papers, ask questions, get summaries, and accelerate your research 🚀</p>",
        unsafe_allow_html=True,
    )

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
    st.subheader("What would you like to do?")
    c1, c2, c3, c4 = st.columns(4)
    nav_cards = [
        (c1, "💬", "Ask Assistant", "Ask questions and get grounded answers.", "Ask Assistant"),
        (c2, "☁️", "Upload & Analyze", "Upload a PDF and index it.", "Upload & Analyze"),
        (c3, "🔍", "Search Papers", "Find relevant passages by meaning.", "Search Papers"),
        (c4, "🧠", "Summarize Text", "Summarize a document or text.", "Summarize Text"),
    ]
    for col, icon2, title, desc, target in nav_cards:
        with col:
            st.markdown(f"<div class='feature-card'><h4>{icon2} {title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
            if st.button("Go →", key=f"card_{target}"):
                st.session_state["page"] = target
                st.rerun()

    st.divider()
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

# ---------- ASK ASSISTANT ----------
elif page == "Ask Assistant":
    st.title("💬 Ask Assistant")
    if not st.session_state["document_text"]:
        st.warning("No document processed yet. Go to **Upload & Analyze** first.")
    else:
        st.caption("Suggested questions:")
        sc = st.columns(len(st.session_state["suggested_questions"]))
        for i, sq in enumerate(st.session_state["suggested_questions"]):
            if sc[i].button(sq):
                st.session_state["question_input"] = sq

        question = st.text_input("Type your research question here...", key="question_input")

        if st.button("Ask", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    result = answer_question(question)
                st.session_state["last_answer"] = {
                    "q": question,
                    "a": result["answer"],
                    "sources": result["sources"],
                }
                db.log_conversation(question, result["answer"], "qa")
            else:
                st.warning("Please type or select a question first.")

        if st.session_state["last_answer"]:
            st.info(st.session_state["last_answer"]["a"])

            with st.expander("📚 View sources"):
                for i, src in enumerate(st.session_state["last_answer"].get("sources", [])):
                    st.markdown(f"**Source {i+1}**")
                    st.caption(src)
                    st.divider()

            if st.button("💾 Save as Note"):
                db.save_note(st.session_state["last_answer"]["q"], st.session_state["last_answer"]["a"])
                st.success("Saved to Notes!")

# ---------- UPLOAD & ANALYZE ----------
elif page == "Upload & Analyze":
    st.title("☁️ Upload & Analyze")
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

            st.success("Document processed! Go to Ask Assistant or Summarize Text.")

# ---------- SEARCH PAPERS ----------
elif page == "Search Papers":
    st.title("🔍 Search Papers")
    st.caption("Semantic search across the currently processed document — no LLM call, just the closest matching passages.")

    if not st.session_state["document_text"]:
        st.warning("No document processed yet. Go to **Upload & Analyze** first.")
    else:
        query = st.text_input("Search for a topic or phrase...")
        if st.button("Search", type="primary") and query:
            results = search(query, top_k=5)
            for i, chunk in enumerate(results):
                st.markdown(f"**Match {i+1}**")
                st.write(chunk)
                st.divider()

# ---------- SUMMARIZE TEXT ----------
elif page == "Summarize Text":
    st.title("🧠 Summarize Text")
    tab1, tab2 = st.tabs(["Summarize uploaded document", "Paste your own text"])

    with tab1:
        if not st.session_state["document_text"]:
            st.warning("No document processed yet. Go to **Upload & Analyze** first.")
        else:
            if st.button("Summarize Document", type="primary"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(st.session_state["document_text"])
                st.info(summary)
                db.log_conversation("(Summary requested)", summary, "summary")

    with tab2:
        custom_text = st.text_area("Paste text to summarize", height=200)
        if st.button("Summarize Pasted Text", type="primary"):
            if custom_text:
                with st.spinner("Summarizing..."):
                    summary = summarize_text(custom_text)
                st.info(summary)
                db.log_conversation("(Custom text summary)", summary, "summary")
            else:
                st.warning("Please paste some text first.")

# ---------- SAVED NOTES ----------
elif page == "Saved Notes":
    st.title("📌 Saved Notes")
    notes = db.get_all_notes()
    if not notes:
        st.caption("No notes saved yet. Save an answer from Ask Assistant to see it here.")
    else:
        for q, a, saved_at in notes:
            with st.expander(f"{q}  ·  saved {db.time_ago(saved_at)}"):
                st.write(a)

# ---------- HISTORY ----------
elif page == "History":
    st.title("🕘 Full History")
    tab1, tab2 = st.tabs(["Conversations", "Uploaded Papers"])

    with tab1:
        convos = db.get_all_conversations()
        if not convos:
            st.caption("No conversations yet.")
        else:
            for q, a, type_, created_at in convos:
                with st.expander(f"[{type_}] {q}  ·  {db.time_ago(created_at)}"):
                    st.write(a)

    with tab2:
        papers = db.get_all_papers()
        if not papers:
            st.caption("No papers uploaded yet.")
        else:
            for filename, size_kb, uploaded_at in papers:
                st.markdown(f"**{filename}**  \n{size_kb} KB · {db.time_ago(uploaded_at)}")