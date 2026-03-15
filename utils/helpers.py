import os
import yt_dlp

from youtube_transcript_api import YouTubeTranscriptApi
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


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


def get_pdf_converter(do_ocr: bool = True, do_table_structure: bool = True):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

def get_youtube_title(video_url: str):
    """Fetches the title of a YouTube video given its URL.
    Returns the title string or an error.
    """
    ydl_opts = {
        "quiet": True,    # Suppress console output
        "skip_download": True,
        "no_warnings": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_title = info_dict.get("title", "Unknown Title")
            return video_title
    except Exception as e:
        return f"Could not fetch video title:\nError -- {str(e)}"
