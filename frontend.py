import uuid
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from state import OutputFormat, OutputConstraints
from backend import app, saver, ingest_youtube, ingest_pdf

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
apply_custom_theme()

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())

with st.sidebar:
    st.markdown('<h1 class="main-title">💎 Lapis</h1>', unsafe_allow_html=True)
    
    st.header("Video Research")
    yt_url = st.text_input("YouTube URL")
    if st.button("Index Video"):
        with st.spinner("Processing..."):
            ingest_youtube(yt_url, st.session_state.current_thread_id)
            st.success("Video context saved for this session!")

    st.header("Document Research")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button("Index document"):
        with st.spinner("Docling-ing..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            num_chunks = ingest_pdf(temp_path, st.session_state.current_thread_id)
            os.remove(temp_path)
            st.success(f"Indexed {num_chunks} chunks from PDF!")
            
    st.divider()
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.current_thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.subheader("Recent Chats")
    threads = list({c.config["configurable"]["thread_id"] for c in saver.list(None)})
    for tid in threads:
        if st.button(f"💬 Session {tid[:8]}", key=tid, use_container_width=True):
            st.session_state.current_thread_id = tid
            st.rerun()

config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
state_snap = app.get_state(config)
history = state_snap.values.get("messages", []) if state_snap.values else []

for msg in history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask about the video..."):
    st.chat_message("user").markdown(prompt)
    inputs = {"messages": [HumanMessage(content=prompt)], 
              "output_format": OutputFormat(name="R1", file_type="pdf", constraints=OutputConstraints())}
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        for event in app.stream(inputs, config=config, stream_mode="values"):
            if "messages" in event:
                full_res = event["messages"][-1].content
        placeholder.markdown(full_res)