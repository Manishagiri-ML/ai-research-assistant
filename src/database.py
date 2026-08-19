import sqlite3
import os
from datetime import datetime

# The database file will sit in your project root, named assistant_data.db
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "assistant_data.db")


def init_db():
    """
    Creates the database file and tables if they don't exist yet.
    Safe to call every time the app starts — never wipes existing data.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            size_kb REAL,
            uploaded_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            type TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_paper(filename, size_kb):
    """Records that a paper was uploaded."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO papers (filename, size_kb, uploaded_at) VALUES (?, ?, ?)",
        (filename, size_kb, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def log_conversation(question, answer, type_):
    """Records a question+answer or a summary. type_ is 'qa' or 'summary'."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (question, answer, type, created_at) VALUES (?, ?, ?, ?)",
        (question, answer, type_, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_stats():
    """Returns real counts pulled from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM papers")
    papers_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM conversations WHERE type='qa'")
    questions_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM conversations WHERE type='summary'")
    summaries_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT date(created_at)) FROM conversations")
    active_days = cursor.fetchone()[0]

    conn.close()
    return {
        "papers": papers_count,
        "questions": questions_count,
        "summaries": summaries_count,
        "active_days": active_days,
    }


def get_recent_papers(limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, size_kb, uploaded_at FROM papers ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_recent_conversations(limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT question, answer, created_at FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def time_ago(timestamp_str):
    """Converts a saved timestamp into a friendly '2 hours ago' style string."""
    then = datetime.fromisoformat(timestamp_str)
    diff = datetime.now() - then
    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    else:
        return f"{int(seconds // 86400)} days ago"