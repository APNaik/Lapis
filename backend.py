import os
from pymongo import MongoClient
from dotenv import load_dotenv

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
    """Query the indexed PDFs and YouTube transcripts for specific details."""
    thread_id = config["configurable"].get("thread_id")
    vector_db_path = get_vector_path(thread_id)
    if not os.path.exists(vector_db_path):
        return "No documents of videos have been indexed yet"
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    relevent_docs = db.similarity_search(query, k=5)
    return "\n\n".join([d.page_content for d in relevent_docs])

tools = [web_search, query_knowledge_base]

# --- Graph Nodes ---
def supervisor_node(state: AgentState):
    """Decides whether to execute a tool or respond to the user"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)
    system_msg = (f"You are Lapis, an expert research agen. Goal: {state.get('research_goal')}."
                  "Use tools to find information if it is not in your context")
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Routing logic based on tool calls"""
    last_message = state["messages"][-1]
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
