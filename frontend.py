import uuid
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from state import OutputFormat, OutputConstraints
from backend import app, saver, ingest_youtube, ingest_pdf

def render_message_content(content):
    """Normalize model message content to a plain string for Streamlit rendering."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif block is not None:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)

def initialize_thread():
    """Retrieves existing thread_id from URL or creates a new one."""
    url_thread_id = st.query_params.get("thread_id")
    
    if url_thread_id:
        # User is returning to an existing session
        st.session_state.current_thread_id = url_thread_id
    elif "current_thread_id" not in st.session_state:
        # First-time visitor: generate ID and update URL
        new_id = str(uuid.uuid4())
        st.session_state.current_thread_id = new_id
        st.query_params["thread_id"] = new_id

def apply_custom_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #050A18; color: #E0E6ED; }
        [data-testid="stSidebar"] { background-color: #0A122A; border-right: 1px solid #1E293B; }
        .stChatMessage { background-color: #111827; border-radius: 10px; border: 1px solid #3B82F6; margin-bottom: 10px; }
        [data-testid="stChatMessage"]:nth-child(even) { background-color: #1E1B4B; border: 1px solid #8B5CF6; }
        .main-title { background: linear-gradient(90deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800; }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Lapis AI", page_icon="💎", layout="wide")
initialize_thread()
apply_custom_theme()

config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
with st.sidebar:
    st.markdown('<h1 class="main-title">💎 Lapis</h1>', unsafe_allow_html=True)
    
    #--- Render current assets---
    st.subheader("Session assets")
    state_snap = app.get_state(config)
    assets = state_snap.values.get("indexed_assets", []) if state_snap.values else []

    if not assets:
        st.caption("No videos indexed yet")
    else:
        for asset in assets:
            if asset["type"] == "video":
                st.markdown(f"Video - {asset['title']}")
            else:
                st.markdown(f"Document - {asset['title']}")

    st.header("Video Research")
    yt_url = st.text_input("YouTube URL")
    if st.button("Index Video"):
        with st.spinner("Processing..."):
            asset_data = ingest_youtube(yt_url, st.session_state.current_thread_id)
            app.update_state(config, {"indexed_assets": [asset_data]})
            st.success(f"Indexed: {asset_data['title']}")
            st.rerun()

    st.header("Document Research")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button("Index document"):
        with st.spinner("Docling-ing..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            asset_data = ingest_pdf(temp_path, st.session_state.current_thread_id, uploaded_file.name)
            os.remove(temp_path)
            app.update_state(config, {"indexed_assets": [asset_data]})
            st.success(f"Indexed file: {uploaded_file.name}")
            st.rerun()
            
    st.divider()
    if st.button("+ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_thread_id = new_id
        st.query_params["thread_id"] = new_id
        st.rerun()
    
    st.subheader("Recent Chats")
    try:
        threads = list({c.config["configurable"]["thread_id"] for c in saver.list(None)})
        for tid in threads:
            if st.button(f"💬 Session {tid[:8]}", key=tid, use_container_width=True):
                st.query_params["thread_id"] = tid
                st.rerun()
    except Exception as e:
        st.caption("History currently unavailable")
        
state_snap = app.get_state(config)
history = state_snap.values.get("messages", []) if state_snap.values else []

for msg in history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(render_message_content(msg.content))

if prompt := st.chat_input("Ask Lapis..."):
    st.chat_message("user").markdown(prompt)
    inputs = {"messages": [HumanMessage(content=prompt)], 
              "output_format": OutputFormat(name="R1", file_type="pdf", constraints=OutputConstraints())}
    
    with st.chat_message("assistant"):
        status_placeholder = st.container()
        answer_placeholder = st.empty()
        full_response = ""

        for chunk in app.stream(inputs, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "supervisor":
                    new_msg = update["messages"][-1]
                    if new_msg.tool_calls:
                        for call in new_msg.tool_calls:
                            with status_placeholder.expander(f"Calling {call['name']}", expanded=False):
                                st.code(f"Input: {call['args']}")
                    else:
                        full_response = render_message_content(new_msg.content)
                        answer_placeholder.markdown(full_response)
                elif node_name == "tools":
                    for msg in update["messages"]:
                        with status_placeholder.expander(f"Completed {msg.name}", expanded=False):
                            tool_output = render_message_content(msg.content).strip()
                            if tool_output:
                                st.markdown(tool_output)
                            else:
                                st.caption("Tool returned no visible output")