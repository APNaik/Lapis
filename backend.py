import os
from pymongo import MongoClient
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from state import AgentState

load_dotenv()


def configure_hf_windows_cache() -> None:
    """Configure Hugging Face caching to avoid Windows symlink-permission failures.

    Why this exists:
    Docling downloads layout/OCR artifacts through ``huggingface_hub`` the first
    time PDF processing runs. On Windows, that library prefers creating symlinks
    inside its cache for deduplication. On machines without Developer Mode,
    Administrator privileges, or an explicit "Create symbolic links" policy,
    symlink creation can fail with ``OSError: [WinError 1314]``. In this project,
    that failure surfaced while calling ``pdf_converter.convert(...)`` during PDF
    ingestion, causing the whole indexing flow to abort.

    What this implementation does:
    - Only runs on Windows.
    - Suppresses the repeated Hugging Face symlink warning noise.
    - Monkey-patches ``huggingface_hub.file_download.are_symlinks_supported`` to
        always report ``False`` on Windows, which forces Hugging Face to use its
        copy/move fallback path instead of attempting symlink creation.

    Why this is safe here:
    - It does not change application behavior on Linux-based deployments such as
        Hugging Face Spaces.
    - It keeps local Windows development working without requiring elevated OS
        privileges.
    - The tradeoff is slightly less efficient cache storage on affected Windows
        machines, which is preferable to PDF ingestion failing entirely.
    """
    if os.name != "nt":
        return

    # os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:
        from huggingface_hub import file_download as hf_file_download
    except Exception:
        return

    original_are_symlinks_supported = hf_file_download.are_symlinks_supported

    def _always_false_on_windows(cache_dir: str) -> bool:
        # Avoid WinError 1314 by forcing the non-symlink code path.
        try:
            original_are_symlinks_supported(cache_dir)
        except Exception:
            pass
        return False

    hf_file_download.are_symlinks_supported = _always_false_on_windows


configure_hf_windows_cache()

# --- Persistence ---
client = MongoClient(os.getenv("MONGODB_URI"))
saver = MongoDBSaver(client, db_name="lapis_db", collection_name="checkpoints", allowed_msgpack_modules=["state"])

# --- RAG Setup ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_transcript(video_url: str):
    video_id = video_url.split("v=")[1].split("&")[0]
    ytt_api = YouTubeTranscriptApi()
    data = ytt_api.fetch(video_id)
    full_text = ""
    timestamp_map = []
    for segment in data:
        start_pos = len(full_text)
        full_text += segment.text + " "
        timestamp_map.append({"start": segment.start, "char_pos": start_pos})
    return full_text, timestamp_map

def ingest_youtube(video_url: str, thread_id: str):
    text, ts_map = get_transcript(video_url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    
    for doc in docs:
        pos = text.find(doc.page_content[:50])
        closest_ts = min(ts_map, key=lambda x: abs(x["char_pos"] - pos))
        doc.metadata["start"] = closest_ts["start"]
        doc.metadata["source"] = video_url

    # Save vector store specifically for this thread
    vector_db_path = f"vector_db/{thread_id}"
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(vector_db_path)
    return len(docs)

# --- Docling setup ---
def get_pdf_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options)
        }
    )
pdf_converter = get_pdf_converter()

def ingest_pdf(pdf_path: str, thread_id: str):
    """Parses scanned/digital PDFs using Docling and saves to thread-specific FAISS."""
    result_md = pdf_converter.convert(pdf_path)
    markdown_content = result_md.document.export_to_markdown()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(markdown_content)
    docs = [
        Document(page_content=chunk, metadata={"source": pdf_path, "type": "pdf"}) for chunk in chunks
    ]

    # Save to the thread specific FAISS
    vector_db_path = f"vector_db/{thread_id}"
    db = FAISS.from_documents(docs, embeddings)
    if os.path.exists(vector_db_path):
        existing_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        existing_db.merge_from(db)
        existing_db.save_local(vector_db_path)
    else:
        db.save_local(vector_db_path)
    
    return len(docs)

# --- Graph Nodes ---
def chatbot_node(state: AgentState, config: RunnableConfig):
    # Strictly using gemini-2.5-flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)
    thread_id = config["configurable"].get("thread_id")
    vector_db_path = f"vector_db/{thread_id}"
    
    context = ""
    # Load persistence-based context if it exists for this thread
    if os.path.exists(vector_db_path):
        db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        last_msg = state["messages"][-1].content
        relevant_docs = db.similarity_search(last_msg, k=3)
        context_parts = []
        for d in relevant_docs:
            if "start" in d.metadata:
                context_parts.append(f"[{d.metadata['start']}s]: {d.page_content}")
            else:
                source = d.metadata.get("source", "unknown")
                context_parts.append(f"[{source}]: {d.page_content}")
        context = "\n".join(context_parts)

    system_prompt = (
        f"You are Lapis. Research Goal: {state.get('research_goal')}. "
        f"If context is provided below, answer based on it and cite sources.\n"
        f"Context:\n{context}"
    )
    
    messages = [AIMessage(content=system_prompt)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
app = workflow.compile(checkpointer=saver)