import json
import os
import re
import shutil
import threading
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from state import LabState, OutputConstraints, OutputFormat
from utils.helpers import configure_hf_windows_cache, get_pdf_converter, get_transcript, get_youtube_title

load_dotenv()
configure_hf_windows_cache()


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
REPORT_DIR = "/data/reports" if os.path.exists("/data") else "reports"
UPLOAD_DIR = "/data/uploads" if os.path.exists("/data") else "uploads"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pdf_converter = get_pdf_converter()
RUNNING_JOBS: dict[str, threading.Thread] = {}


def lab_config(lab_id: str) -> dict:
    return {"configurable": {"thread_id": lab_id}}


def get_vector_path(lab_id: str) -> str:
    path = os.path.join(PERSISTENT_DIR, lab_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_upload_path(lab_id: str, filename: str) -> str:
    safe_name = sanitize_filename(filename)
    path = os.path.join(UPLOAD_DIR, lab_id, safe_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "lapis"


def lab_node(state: LabState) -> dict:
    return {}


workflow = StateGraph(LabState)
workflow.add_node("lab", lab_node)
workflow.add_edge(START, "lab")
workflow.add_edge("lab", END)
app = workflow.compile(checkpointer=saver)


def get_lab_state(lab_id: str) -> dict:
    snapshot = app.get_state(lab_config(lab_id))
    return snapshot.values if snapshot and snapshot.values else {}


def update_lab_state(lab_id: str, values: dict) -> None:
    app.update_state(lab_config(lab_id), values, as_node="lab")


def list_labs() -> list[dict]:
    labs = []
    seen = set()
    try:
        checkpoints = list(saver.list(None))
    except Exception:
        return []

    for checkpoint in checkpoints:
        lab_id = checkpoint.config["configurable"].get("thread_id")
        if not lab_id or lab_id in seen:
            continue
        seen.add(lab_id)
        state = get_lab_state(lab_id)
        if state.get("lab_id") != lab_id:
            continue
        labs.append(
            {
                "lab_id": lab_id,
                "lab_title": state.get("lab_title") or f"Lab {lab_id[:8]}",
                "status": state.get("status", "created"),
            }
        )
    return sorted(labs, key=lambda item: item["lab_title"].lower())


def create_lab(
    lab_id: str,
    lab_title: str,
    research_goal: str,
    output_constraints: OutputConstraints,
) -> None:
    update_lab_state(
        lab_id,
        {
            "lab_id": lab_id,
            "lab_title": lab_title.strip() or f"Lab {lab_id[:8]}",
            "research_goal": research_goal.strip(),
            "status": "created",
            "status_message": "Lab created. Add seed sources, then start the autonomous run.",
            "output_format": OutputFormat(constraints=output_constraints),
            "confidence": 0.0,
        },
    )


def delete_lab(lab_id: str) -> None:
    try:
        saver.delete_thread(lab_id)
    except Exception:
        pass

    for base_dir in (PERSISTENT_DIR, REPORT_DIR, UPLOAD_DIR):
        path = os.path.join(base_dir, lab_id)
        if os.path.exists(path):
            shutil.rmtree(path)


def add_documents_to_lab(lab_id: str, docs: list[Document]) -> None:
    if not docs:
        return

    vector_db_path = get_vector_path(lab_id)
    db = FAISS.from_documents(docs, embeddings)
    index_file = os.path.join(vector_db_path, "index.faiss")
    if os.path.exists(index_file):
        existing_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        existing_db.merge_from(db)
        existing_db.save_local(vector_db_path)
    else:
        db.save_local(vector_db_path)


def ingest_youtube(video_url: str, lab_id: str) -> dict:
    text, ts_map = get_transcript(video_url)
    video_title = get_youtube_title(video_url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    for doc in docs:
        pos = text.find(doc.page_content[:50])
        closest_ts = min(ts_map, key=lambda x: abs(x["char_pos"] - pos)) if ts_map else {"start": 0}
        doc.metadata["start"] = closest_ts["start"]
        doc.metadata["source"] = video_url
        doc.metadata["type"] = "video"
        doc.metadata["title"] = video_title

    add_documents_to_lab(lab_id, docs)
    asset = {"title": video_title, "type": "video", "source": video_url}
    update_lab_state(lab_id, {"indexed_assets": [asset], "seed_sources": [asset]})
    return asset


def ingest_pdf(pdf_path: str, lab_id: str, name_pdf: str) -> dict:
    result_md = pdf_converter.convert(pdf_path)
    markdown_content = result_md.document.export_to_markdown()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(markdown_content)
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": pdf_path, "type": "pdf", "title": name_pdf},
        )
        for chunk in chunks
    ]

    add_documents_to_lab(lab_id, docs)
    asset = {"title": name_pdf, "type": "pdf", "source": pdf_path}
    update_lab_state(lab_id, {"indexed_assets": [asset], "seed_sources": [asset]})
    return asset


def query_lab_knowledge(lab_id: str, query: str, k: int = 10) -> list[Document]:
    vector_db_path = get_vector_path(lab_id)
    index_file = os.path.join(vector_db_path, "index.faiss")
    if not os.path.exists(index_file):
        return []

    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db.similarity_search(query, k=k)


def invoke_llm(prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def derive_search_queries(research_goal: str, asset_titles: list[str]) -> list[str]:
    fallback = [
        research_goal,
        f"{research_goal} recent research",
        f"{research_goal} review paper",
        f"{research_goal} evidence and limitations",
    ]
    prompt = (
        "Return JSON only. Create 4 concise web search queries for a research lab.\n"
        f"Research goal: {research_goal}\n"
        f"Seed materials: {', '.join(asset_titles) if asset_titles else 'none'}\n"
        'Schema: {"queries": ["query 1", "query 2"]}'
    )
    try:
        raw = invoke_llm(prompt)
        parsed = json.loads(extract_json(raw))
        queries = [str(item).strip() for item in parsed.get("queries", []) if str(item).strip()]
        return dedupe_strings(queries + fallback)[:5]
    except Exception:
        return dedupe_strings(fallback)[:5]


def extract_json(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value).strip()
        value = re.sub(r"```$", "", value).strip()
    match = re.search(r"\{.*\}", value, re.DOTALL)
    return match.group(0) if match else value


def dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def discover_sources(research_goal: str, existing_urls: set[str], asset_titles: list[str]) -> list[dict]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    search = TavilySearch(max_results=5)
    queries = derive_search_queries(research_goal, asset_titles)
    discovered = []

    for query in queries:
        response = search.invoke({"query": query})
        results = normalize_tavily_results(response)
        for result in results:
            url = result.get("url")
            if not url or url in existing_urls:
                continue
            existing_urls.add(url)
            discovered.append(
                {
                    "title": result.get("title") or url,
                    "url": url,
                    "snippet": result.get("content") or result.get("snippet") or "",
                    "query": query,
                    "relevance_reason": f"Found while researching: {query}",
                    "confidence": float(result.get("score") or 0.6),
                }
            )
    return discovered[:10]


def normalize_tavily_results(response: Any) -> list[dict]:
    if isinstance(response, dict):
        results = response.get("results", [])
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    return []


def ingest_web_sources(lab_id: str, sources: list[dict]) -> int:
    docs = []
    for source in sources:
        snippet = source.get("snippet", "").strip()
        if len(snippet) < 40:
            continue
        docs.append(
            Document(
                page_content=f"{source.get('title', 'Untitled')}\n\n{snippet}",
                metadata={
                    "source": source.get("url"),
                    "type": "web",
                    "title": source.get("title", "Untitled"),
                },
            )
        )
    add_documents_to_lab(lab_id, docs)
    return len(docs)


def calculate_confidence(indexed_assets: list[dict], discovered_sources: list[dict], ingested_web_count: int) -> float:
    score = 0.25 if indexed_assets else 0.0
    score += min(len(discovered_sources), 6) * 0.08
    score += min(ingested_web_count, 6) * 0.045
    return round(min(score, 0.95), 2)


def run_lab(lab_id: str) -> dict:
    state = get_lab_state(lab_id)
    research_goal = state.get("research_goal", "").strip()
    if not research_goal:
        update_lab_state(
            lab_id,
            {
                "status": "failed",
                "error": "Research goal is required before starting a lab run.",
                "status_message": "Research goal is missing.",
            },
        )
        return get_lab_state(lab_id)

    indexed_assets = state.get("indexed_assets", [])
    asset_titles = [asset.get("title", "") for asset in indexed_assets if asset.get("title")]
    existing_urls = {asset.get("source") for asset in indexed_assets if asset.get("source")}
    existing_urls.update(
        source.get("url") for source in state.get("discovered_sources", []) if source.get("url")
    )

    try:
        update_lab_state(lab_id, {"status": "researching", "status_message": "Searching for related sources."})
        discovered = discover_sources(research_goal, existing_urls, asset_titles)
        ingested_web_count = ingest_web_sources(lab_id, discovered)
        confidence = calculate_confidence(indexed_assets, discovered, ingested_web_count)
        notes = [
            f"Discovered {len(discovered)} sources with Tavily.",
            f"Ingested {ingested_web_count} web snippets into the lab knowledge base.",
            f"Confidence score: {confidence}",
        ]
        update_lab_state(
            lab_id,
            {
                "discovered_sources": discovered,
                "research_notes": notes,
                "confidence": confidence,
            },
        )

        if confidence < 0.55:
            request = (
                "Please add one authoritative PDF, paper title, or reference URL for this topic. "
                "The current source coverage is too thin for an autonomous report."
            )
            update_lab_state(
                lab_id,
                {
                    "status": "needs_input",
                    "status_message": request,
                    "requested_resources": [request],
                },
            )
            return get_lab_state(lab_id)

        generate_lab_report(lab_id)
        return get_lab_state(lab_id)
    except Exception as exc:
        update_lab_state(
            lab_id,
            {
                "status": "failed",
                "error": str(exc),
                "status_message": "The lab run failed. Check configuration and source availability.",
            },
        )
        return get_lab_state(lab_id)


def start_lab_run(lab_id: str) -> bool:
    existing_job = RUNNING_JOBS.get(lab_id)
    if existing_job and existing_job.is_alive():
        return False

    job = threading.Thread(target=run_lab, args=(lab_id,), daemon=True)
    RUNNING_JOBS[lab_id] = job
    job.start()
    return True


def is_lab_running(lab_id: str) -> bool:
    job = RUNNING_JOBS.get(lab_id)
    return bool(job and job.is_alive())


def generate_lab_report(lab_id: str) -> dict:
    state = get_lab_state(lab_id)
    research_goal = state.get("research_goal", "")
    output_format = state.get("output_format") or OutputFormat(constraints=OutputConstraints())
    if isinstance(output_format, dict):
        output_format = OutputFormat(**output_format)

    update_lab_state(lab_id, {"status": "drafting", "status_message": "Drafting the PDF report."})
    target_words = (output_format.constraints.pages or 1) * (output_format.constraints.words_per_page or 300)
    docs = query_lab_knowledge(lab_id, research_goal, k=12)
    context = format_context(docs)
    sources = state.get("discovered_sources", []) + state.get("indexed_assets", [])

    prompt = (
        "Write a concise research report in Markdown.\n"
        f"Research goal: {research_goal}\n"
        f"Target length: about {target_words} words.\n"
        "Use clear section headings. Include a final 'Sources' section with URLs or file names.\n\n"
        f"Context:\n{context}\n\n"
        f"Known sources:\n{json.dumps(sources[:16], indent=2)}"
    )
    try:
        draft = invoke_llm(prompt)
    except Exception:
        draft = build_fallback_report(research_goal, docs, sources, target_words)

    report_path = render_pdf_report(lab_id, state.get("lab_title") or "Lapis Lab Report", draft)
    update_lab_state(
        lab_id,
        {
            "draft": draft,
            "report_path": report_path,
            "status": "complete",
            "status_message": "Report generated.",
        },
    )
    return get_lab_state(lab_id)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "Untitled")
        content = doc.page_content[:1800]
        blocks.append(f"[{index}] {title}\nSource: {source}\n{content}")
    return "\n\n".join(blocks) or "No indexed context available."


def build_fallback_report(
    research_goal: str,
    docs: list[Document],
    sources: list[dict],
    target_words: int,
) -> str:
    excerpts = "\n\n".join(doc.page_content[:700] for doc in docs[:6])
    source_lines = []
    for source in sources[:12]:
        label = source.get("title") or source.get("source") or source.get("url") or "Source"
        link = source.get("url") or source.get("source") or ""
        source_lines.append(f"- {label}: {link}")

    return (
        f"# Research Report\n\n"
        f"## Goal\n\n{research_goal}\n\n"
        f"## Summary\n\n"
        f"This report was generated from the indexed lab materials. The target length was "
        f"approximately {target_words} words, but the language model was unavailable, so this "
        f"fallback draft focuses on retrieved excerpts.\n\n"
        f"## Retrieved Evidence\n\n{excerpts or 'No retrieved evidence was available.'}\n\n"
        f"## Sources\n\n" + "\n".join(source_lines)
    )


def render_pdf_report(lab_id: str, title: str, markdown_text: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    report_dir = os.path.join(REPORT_DIR, lab_id)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{sanitize_filename(title)}.pdf")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_path, pagesize=letter, title=title)
    story = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape_pdf_text(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape_pdf_text(line[3:]), styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(escape_pdf_text(line[4:]), styles["Heading3"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"- {escape_pdf_text(line[2:])}", styles["BodyText"]))
        else:
            story.append(Paragraph(escape_pdf_text(line), styles["BodyText"]))
    doc.build(story)
    return report_path


def escape_pdf_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
