import os
from pymongo import MongoClient
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

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
        context = "\n".join([f"[{d.metadata['start']}s]: {d.page_content}" for d in relevant_docs])

    system_prompt = (
        f"You are Lapis. Research Goal: {state.get('research_goal')}. "
        f"If context is provided below, answer based on it and cite timestamps.\n"
        f"Context from Video:\n{context}"
    )
    
    messages = [AIMessage(content=system_prompt)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
app = workflow.compile(checkpointer=saver)