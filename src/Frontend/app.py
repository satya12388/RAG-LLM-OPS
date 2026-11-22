import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Ensure the project root and src are on sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# Load environment variables directly from .env file
load_dotenv()

from src.Frontend.api_client import ApiClient

SESSION_FILE = Path(os.getenv("SESSIONS_FILE_PATH")) if os.getenv("SESSIONS_FILE_PATH") else (ROOT_PATH / "frontend_session.json")


# -----------------------------------------------------------------------------
# Storage Helpers for frontend_session.json
# -----------------------------------------------------------------------------
def load_sessions() -> Dict[str, Any]:
    """Load all sessions from frontend_session.json in the root folder."""
    if not SESSION_FILE.exists():
        return {}
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        st.sidebar.error(f"Error loading sessions: {e}")
        return {}


def save_sessions(sessions: Dict[str, Any]) -> None:
    """Save sessions to frontend_session.json in the configured path."""
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving session: {e}")


# -----------------------------------------------------------------------------
# LLM Session Renamer using ChatGroq
# -----------------------------------------------------------------------------
def rename_session_with_llm(messages: List[Dict[str, str]]) -> str:
    """
    Use ChatGroq to generate a concise 3-5 word title for the session based on messages.
    Uses GROQ_API_KEY from .env loaded via load_dotenv.
    Falls back gracefully if the API key is missing or an error occurs.
    """
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
    )
    convo_snippet = "\n".join(
        [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')[:250]}" for m in messages[:4]]
    )
    prompt = (
        "Generate a concise, descriptive title (3 to 5 words maximum) for this conversation. "
        "Do NOT include quotes, bullet points, asterisks, or trailing punctuation. "
        "Return ONLY the plain title text:\n\n"
        f"{convo_snippet}"
    )
    resp = llm.invoke([
        SystemMessage(content="You generate short, clean, descriptive 3-5 word conversation titles."),
        HumanMessage(content=prompt),
    ])
    title = resp.content.strip().strip('"\'*#')
    title = title.split("\n")[0].strip()
    return title if title else "Chat Session"
    


# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="RAG Graph Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Styling for clean and simple UI
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        .stChatMessage {
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
        }
        .sidebar-header {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_sessions()

    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")

    # Ensure at least one session exists
    if not st.session_state.sessions:
        initial_key = "Chat 1"
        st.session_state.sessions[initial_key] = {
            "title": initial_key,
            "created_at": datetime.now().isoformat(),
            "messages": [],
        }
        save_sessions(st.session_state.sessions)
        st.session_state.current_session_key = initial_key

    if "current_session_key" not in st.session_state or st.session_state.current_session_key not in st.session_state.sessions:
        st.session_state.current_session_key = list(st.session_state.sessions.keys())[0]

    client = ApiClient(base_url=st.session_state.api_url)

    # -----------------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------------
    with st.sidebar:
        st.title("💬 Chat Sessions")

        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            count = len(st.session_state.sessions) + 1
            new_key = f"Chat {count}"
            while new_key in st.session_state.sessions:
                count += 1
                new_key = f"Chat {count}"

            st.session_state.sessions[new_key] = {
                "title": new_key,
                "created_at": datetime.now().isoformat(),
                "messages": [],
            }
            save_sessions(st.session_state.sessions)
            st.session_state.current_session_key = new_key
            st.rerun()

        st.markdown("---")

        # List of sessions
        session_keys = list(st.session_state.sessions.keys())
        st.markdown("**History**")

        for key in session_keys:
            col1, col2 = st.columns([0.82, 0.18])
            is_active = key == st.session_state.current_session_key
            btn_label = f"● {key}" if is_active else f"  {key}"

            with col1:
                if st.button(btn_label, key=f"session_btn_{key}", use_container_width=True):
                    st.session_state.current_session_key = key
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"del_{key}", help="Delete session"):
                    del st.session_state.sessions[key]
                    if not st.session_state.sessions:
                        new_key = "Chat 1"
                        st.session_state.sessions[new_key] = {
                            "title": new_key,
                            "created_at": datetime.now().isoformat(),
                            "messages": [],
                        }
                        st.session_state.current_session_key = new_key
                    elif st.session_state.current_session_key == key:
                        st.session_state.current_session_key = list(st.session_state.sessions.keys())[0]
                    save_sessions(st.session_state.sessions)
                    st.rerun()

        st.markdown("---")

        # Settings & Status
        with st.expander("🔧 Settings & Status", expanded=False):
            backend_url = st.text_input("Backend API URL", value=st.session_state.api_url)
            if backend_url != st.session_state.api_url:
                st.session_state.api_url = backend_url

            health = client.check_health()
            if health.get("status") == "ok":
                st.success("🟢 Backend: Connected")
            else:
                st.error("🔴 Backend: Offline / Unreachable")
                st.caption(health.get("message", ""))

    # -----------------------------------------------------------------------------
    # Main Chat Area
    # -----------------------------------------------------------------------------
    current_key = st.session_state.current_session_key
    session_data = st.session_state.sessions[current_key]
    messages = session_data.setdefault("messages", [])

    # Title header
    col_title, col_status = st.columns([0.8, 0.2])
    with col_title:
        st.subheader(f"💬 {session_data.get('title', current_key)}")
    with col_status:
        st.caption(f"{len(messages)} messages")

    # Display conversation messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User prompt input
    if prompt := st.chat_input("Ask a question..."):
        # 1. Append & display user message
        user_msg = {"role": "user", "content": prompt}
        messages.append(user_msg)
        save_sessions(st.session_state.sessions)

        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Stream assistant response
        with st.chat_message("assistant"):
            stream_gen = client.stream_chat(query=prompt, session_id=current_key)
            full_response = st.write_stream(stream_gen)

        # 3. Save assistant response
        ai_msg = {"role": "assistant", "content": full_response or ""}
        messages.append(ai_msg)
        save_sessions(st.session_state.sessions)

        # 4. Auto-rename the session once the first AI response arrives (no manual trigger needed)
        is_first_exchange = (len(messages) == 2)
        is_generic_title = current_key.startswith("Chat ") or current_key == "New Chat"

        if is_first_exchange or is_generic_title:
            new_title = rename_session_with_llm(messages)
            if new_title and new_title != current_key:
                final_title = new_title
                counter = 1
                while final_title in st.session_state.sessions and final_title != current_key:
                    final_title = f"{new_title} ({counter})"
                    counter += 1

                session_content = st.session_state.sessions.pop(current_key)
                session_content["title"] = final_title
                session_content["updated_at"] = datetime.now().isoformat()
                st.session_state.sessions[final_title] = session_content
                st.session_state.current_session_key = final_title
                save_sessions(st.session_state.sessions)
                st.rerun()


if __name__ == "__main__":
    main()
