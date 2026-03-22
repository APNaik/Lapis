import os
from pymongo import MongoClient
from dotenv import load_dotenv
import json

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import ToolNode

from state import AgentState
from utils.helpers import configure_hf_windows_cache, get_pdf_converter, get_transcript, get_youtube_title

load_dotenv()
configure_hf_windows_cache()

# --- Persistence ---
client = MongoClient(os.getenv("MONGODB_URI"))
ALLOWED_MSGPACK_MODULES = [
    ("state", "OutputFormat"),
    ("state", "OutputConstraints"),
]
saver = MongoDBSaver(
    client,
    db_name="lapis_db",
    collection_name="checkpoints",
    serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES),
)

PERSISTENT_DIR = "/data/vector_db" if os.path.exists("/data") else "vector_db"

def get_vector_path(thread_id: str):
    path = os.path.join(PERSISTENT_DIR, thread_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

# --- RAG Setup ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_youtube(video_url: str, thread_id: str):
    text, ts_map = get_transcript(video_url)

    video_title = get_youtube_title(video_url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    
    for doc in docs:
        pos = text.find(doc.page_content[:50])
        closest_ts = min(ts_map, key=lambda x: abs(x["char_pos"] - pos))
        doc.metadata["start"] = closest_ts["start"]
        doc.metadata["source"] = video_url

    # Save vector store specifically for this thread
    vector_db_path = get_vector_path(thread_id)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(vector_db_path)

    return {
        "title": video_title,
        "type": "video",
        "source": video_url
    }

# --- Docling setup ---
pdf_converter = get_pdf_converter()
 
def ingest_pdf(pdf_path: str, thread_id: str, name_pdf: str):
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
    vector_db_path = get_vector_path(thread_id)
    db = FAISS.from_documents(docs, embeddings)
    index_file = os.path.join(vector_db_path, "index.faiss")
    if os.path.exists(index_file):
        existing_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        existing_db.merge_from(db)
        existing_db.save_local(vector_db_path)
    else:
        db.save_local(vector_db_path)
    
    return {
        "title": name_pdf,
        "type": "pdf",
        "source": pdf_path
    }

# -- Tools setup ---
@tool
def web_search(query: str):
    """Search the web using Tavily for real-time information and links"""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return (
            "Web search is not configured. Set TAVILY_API_KEY in your .env file "
            "and restart the Streamlit app."
        )

    try:
        search = TavilySearch(max_results=5)
        return search.invoke({"query": query})
    except Exception as e:
        return (
            "Web search failed. Verify TAVILY_API_KEY is valid and Tavily service is reachable. "
            f"Details: {str(e)}"
        )

@tool
def query_knowledge_base(query: str, config: RunnableConfig):
    """
    USE THIS TOOL FIRST for any questions about videos or PDFs.
    This tool searches the INTERNAL database of transcripts and documents 
    that the user has already indexed in this session.
    Input should be a specific search query based on the user's question.
    """
    thread_id = config["configurable"].get("thread_id")
    vector_db_path = get_vector_path(thread_id)
    if not os.path.exists(vector_db_path):
        return "No documents of videos have been indexed yet"
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    relevent_docs = db.similarity_search(query, k=5)
    results = [{"content": d.page_content, "source": d.metadata.get("source")} for d in relevent_docs]
    return json.dumps(results)

tools = [web_search, query_knowledge_base]

# --- Graph Nodes ---
def supervisor_node(state: AgentState):
    """Decides whether to execute a tool or respond to the user"""
    assets = state.get("indexed_assets", [])
    asset_titles = [a['title'] for a in assets]

    asset_context = ""
    if asset_titles:
        asset_context = f"\nYou have access to the following indexed materials: {', '.join(asset_titles)}."
        asset_context += "\nTo answer questions about these, you MUST use the 'query_knowledge_base' tool."
    else:
        asset_context = "\nNo documents or videos have been indexed yet. Tell the user to upload/index them if they ask questions."
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite").bind_tools(tools)

    system_msg = (
        f"You are Lapis, a research assistant. Goal: {state.get('research_goal', 'Assist user')}."
        f"{asset_context}"
        "\nDo not ask the user for links if the asset is already listed above; use your tools instead."
    )
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Routing logic based on tool calls"""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", should_continue)
workflow.add_edge("tools", "supervisor")
app = workflow.compile(checkpointer=saver)
