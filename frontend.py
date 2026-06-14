import os
import uuid

import streamlit as st

from backend import (
    create_lab,
    delete_lab,
    generate_lab_report,
    get_lab_state,
    get_upload_path,
    ingest_pdf,
    ingest_youtube,
    is_lab_running,
    list_labs,
    start_lab_run,
)
from state import OutputConstraints


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #08111f; color: #e5edf7; }
        [data-testid="stSidebar"] { background-color: #0d1728; border-right: 1px solid #243244; }
        .main-title {
            color: #dbeafe;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .status-box {
            border: 1px solid #334155;
            background: #101c2f;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_lab_selection() -> None:
    url_lab_id = st.query_params.get("lab_id")
    if url_lab_id:
        st.session_state.current_lab_id = url_lab_id
    elif "current_lab_id" not in st.session_state:
        labs = list_labs()
        if labs:
            st.session_state.current_lab_id = labs[0]["lab_id"]
            st.query_params["lab_id"] = labs[0]["lab_id"]
        else:
            st.session_state.current_lab_id = None


def set_current_lab(lab_id: str) -> None:
    st.session_state.current_lab_id = lab_id
    st.query_params["lab_id"] = lab_id


def render_lab_creator() -> None:
    with st.form("create_lab_form"):
        st.subheader("New Lab")
        lab_title = st.text_input("Lab title", placeholder="Example: Multimodal RAG evaluation")
        research_goal = st.text_area(
            "Research goal",
            placeholder="Define what the lab should learn, compare, or explain.",
            height=120,
        )
        col1, col2 = st.columns(2)
        with col1:
            pages = st.number_input("Pages", min_value=1, max_value=25, value=3, step=1)
        with col2:
            words_per_page = st.number_input(
                "Words/page",
                min_value=150,
                max_value=800,
                value=300,
                step=50,
            )

        submitted = st.form_submit_button("Create lab", use_container_width=True)
        if submitted:
            if not research_goal.strip():
                st.error("Research goal is required.")
                return
            lab_id = str(uuid.uuid4())
            create_lab(
                lab_id=lab_id,
                lab_title=lab_title,
                research_goal=research_goal,
                output_constraints=OutputConstraints(
                    pages=int(pages),
                    words_per_page=int(words_per_page),
                ),
            )
            set_current_lab(lab_id)
            st.rerun()


def render_lab_list() -> None:
    st.subheader("Labs")
    labs = list_labs()
    if not labs:
        st.caption("No labs yet")
        return

    current_lab_id = st.session_state.get("current_lab_id")
    for lab in labs:
        label = f"{lab['lab_title']} · {lab['status']}"
        button_type = "primary" if lab["lab_id"] == current_lab_id else "secondary"
        if st.button(label, key=f"select_{lab['lab_id']}", use_container_width=True, type=button_type):
            set_current_lab(lab["lab_id"])
            st.rerun()


def render_seed_controls(lab_id: str) -> None:
    st.subheader("Seed Sources")
    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Add video seed", use_container_width=True):
        if not yt_url.strip():
            st.error("Enter a YouTube URL first.")
        else:
            with st.spinner("Indexing video transcript..."):
                try:
                    asset = ingest_youtube(yt_url.strip(), lab_id)
                    st.success(f"Indexed video: {asset['title']}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Video ingestion failed: {exc}")

    uploaded_file = st.file_uploader("Upload research PDF", type=["pdf"])
    if uploaded_file and st.button("Add PDF seed", use_container_width=True):
        with st.spinner("Parsing and indexing PDF..."):
            try:
                upload_path = get_upload_path(lab_id, uploaded_file.name)
                with open(upload_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())
                asset = ingest_pdf(upload_path, lab_id, uploaded_file.name)
                st.success(f"Indexed PDF: {asset['title']}")
                st.rerun()
            except Exception as exc:
                st.error(f"PDF ingestion failed: {exc}")


def render_assets(state: dict) -> None:
    assets = state.get("indexed_assets", [])
    if not assets:
        st.caption("No seed sources indexed yet.")
        return

    for asset in assets:
        label = "Video" if asset.get("type") == "video" else "Document"
        st.markdown(f"**{label}:** {asset.get('title', 'Untitled')}")


def render_sources(state: dict) -> None:
    sources = state.get("discovered_sources", [])
    if not sources:
        st.caption("No discovered web sources yet.")
        return

    for source in sources:
        title = source.get("title") or source.get("url") or "Source"
        url = source.get("url", "")
        confidence = source.get("confidence", 0)
        st.markdown(f"**[{title}]({url})**")
        st.caption(f"Query: {source.get('query', 'n/a')} · Confidence: {confidence}")
        snippet = source.get("snippet")
        if snippet:
            st.write(snippet)
        st.divider()


def render_report_download(state: dict) -> None:
    report_path = state.get("report_path")
    if not report_path or not os.path.exists(report_path):
        return

    with open(report_path, "rb") as report_file:
        st.download_button(
            "Download PDF report",
            data=report_file,
            file_name=os.path.basename(report_path),
            mime="application/pdf",
            use_container_width=True,
        )


def render_lab_workspace(lab_id: str) -> None:
    state = get_lab_state(lab_id)
    if not state:
        st.warning("This lab could not be loaded.")
        return

    st.markdown(f"# {state.get('lab_title', 'Untitled Lab')}")
    st.write(state.get("research_goal", ""))

    status = state.get("status", "created")
    confidence = state.get("confidence", 0.0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status)
    col2.metric("Confidence", f"{confidence:.2f}")
    output_format = state.get("output_format", {})
    if isinstance(output_format, dict):
        constraints = output_format.get("constraints", {})
    else:
        constraints = getattr(output_format, "constraints", OutputConstraints())
    if isinstance(constraints, dict):
        pages = constraints.get("pages", 1)
        words_per_page = constraints.get("words_per_page", 300)
    else:
        pages = getattr(constraints, "pages", 1)
        words_per_page = getattr(constraints, "words_per_page", 300)
    col3.metric("Target", f"{pages}p / {words_per_page} wpp")

    message = state.get("status_message")
    if message:
        st.markdown(f"<div class='status-box'>{message}</div>", unsafe_allow_html=True)

    running = is_lab_running(lab_id)
    if running:
        st.info("Autonomous research is running in the background.")

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("Start autonomous run", use_container_width=True, type="primary", disabled=running):
            started = start_lab_run(lab_id)
            if not started:
                st.info("This lab is already running.")
            st.rerun()
    with action_col2:
        if st.button("Generate PDF from current sources", use_container_width=True, disabled=running):
            with st.spinner("Generating report..."):
                generate_lab_report(lab_id)
            st.rerun()
    with action_col3:
        if st.button("Refresh status", use_container_width=True):
            st.rerun()

    if status == "needs_input":
        st.warning(state.get("requested_resources", ["Additional resource required."])[-1])

    render_report_download(state)

    tab_sources, tab_discovery, tab_notes, tab_report = st.tabs(
        ["Seed Sources", "Discovered Sources", "Run Notes", "Draft"]
    )
    with tab_sources:
        render_seed_controls(lab_id)
        st.divider()
        render_assets(state)
    with tab_discovery:
        render_sources(state)
    with tab_notes:
        notes = state.get("research_notes", [])
        if notes:
            for note in notes:
                st.markdown(f"- {note}")
        else:
            st.caption("No run notes yet.")
        error = state.get("error")
        if error:
            st.error(error)
    with tab_report:
        draft = state.get("draft")
        if draft:
            st.markdown(draft)
        else:
            st.caption("No draft generated yet.")

    st.divider()
    if st.button("Delete this lab", use_container_width=True, type="secondary"):
        st.session_state.confirm_delete_lab = True
    if st.session_state.get("confirm_delete_lab"):
        st.warning("This will remove the lab state, vector index, uploads, and generated reports.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm delete", use_container_width=True):
                delete_lab(lab_id)
                st.session_state.current_lab_id = None
                st.session_state.confirm_delete_lab = False
                st.query_params.clear()
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete_lab = False
                st.rerun()


st.set_page_config(page_title="Lapis Labs", page_icon="💎", layout="wide")
apply_custom_theme()
initialize_lab_selection()

with st.sidebar:
    st.markdown('<div class="main-title">Lapis Labs</div>', unsafe_allow_html=True)
    render_lab_creator()
    st.divider()
    render_lab_list()

current_lab_id = st.session_state.get("current_lab_id")
if current_lab_id:
    render_lab_workspace(current_lab_id)
else:
    st.markdown("# Create a lab")
    st.write("Set a research goal, choose the target PDF size, and add seed sources to begin.")
