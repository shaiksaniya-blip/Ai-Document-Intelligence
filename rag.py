import os
import uuid
import logging
import requests

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Document chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

# RAG
RAG_TOP_K = 3

# Lower generation = faster CPU response
MAX_TOKENS = 60

# Keep context small for faster CPU processing
MAX_CONTEXT_CHARS = 2500

# Ollama context window
OLLAMA_CONTEXT = 1536

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(path="./chroma_data")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

documents_collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(file_path: str) -> str:
    """
    Extract text from PDF, DOCX or TXT.
    """

    extension = os.path.splitext(file_path)[1].lower()

    try:

        if extension == ".pdf":

            reader = PdfReader(file_path)

            pages = []

            for page in reader.pages:
                text = page.extract_text() or ""

                if text.strip():
                    pages.append(text)

            return "\n".join(pages).strip()

        elif extension == ".docx":

            document = Document(file_path)

            paragraphs = []

            for paragraph in document.paragraphs:

                text = paragraph.text.strip()

                if text:
                    paragraphs.append(text)

            return "\n".join(paragraphs).strip()

        elif extension == ".txt":

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                return file.read().strip()

        else:

            raise ValueError(
                f"Unsupported document type: {extension}"
            )

    except Exception as e:

        logger.exception("Text extraction failed")
        raise e


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):
    """
    Split text into overlapping chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


# ============================================================
# STORE DOCUMENT
# ============================================================

def store_document_chunks(filename: str, text: str):
    """
    Delete old chunks for the same filename
    and store new chunks.
    """

    filename = filename.strip()

    if not filename:
        raise ValueError(
            "Filename cannot be empty."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "No readable text found in document."
        )

    try:

        documents_collection.delete(
            where={"source": filename}
        )

    except Exception:

        logger.warning(
            "Could not delete previous chunks for %s",
            filename
        )

    ids = []
    metadatas = []

    for index in range(len(chunks)):

        ids.append(
            f"{filename}_{uuid.uuid4().hex}"
        )

        metadatas.append(
            {
                "source": filename,
                "chunk_index": index
            }
        )

    documents_collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    logger.info(
        "Stored %d chunks for %s",
        len(chunks),
        filename
    )

    return {
        "filename": filename,
        "chunks": len(chunks)
    }


# ============================================================
# LIMIT CONTEXT
# ============================================================

def limit_context(
    text: str,
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    Limit text sent to Ollama.
    """

    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars].rstrip()
        + "\n...[content shortened]"
    )


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt: str) -> str:
    """
    Send a prompt to local Ollama.

    Optimized for CPU performance.
    """

    try:

        response = requests.post(
            OLLAMA_URL,

            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,

                "options": {
                    "temperature": 0.1,

                    # Short answers generate much faster
                    "num_predict": MAX_TOKENS,

                    # Smaller context = less CPU work
                    "num_ctx": OLLAMA_CONTEXT,

                    # CPU-friendly
                    "top_k": 20,
                    "top_p": 0.8
                },

                # Keep model loaded between requests
                "keep_alive": "30m"
            },

            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        if not answer:

            return (
                "⚠️ Ollama returned an empty response."
            )

        return answer

    except requests.exceptions.Timeout:

        logger.error(
            "Ollama request timed out."
        )

        return (
            "⚠️ Ollama took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        logger.error(
            "Could not connect to Ollama."
        )

        return (
            "⚠️ Could not connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except requests.exceptions.RequestException as e:

        logger.exception(
            "Ollama request failed."
        )

        return (
            f"⚠️ Ollama request failed: {e}"
        )

    except Exception as e:

        logger.exception(
            "Unexpected Ollama error."
        )

        return (
            f"⚠️ AI generation failed: {e}"
        )


# ============================================================
# CHATBOT / RAG
# ============================================================

def answer_question(
    question: str,
    filename: str = None
) -> dict:
    """
    Answer a question using relevant document chunks.
    """

    question = question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    try:

        # ----------------------------------------------------
        # SEARCH DOCUMENT
        # ----------------------------------------------------

        if filename:

            results = documents_collection.query(
                query_texts=[question],
                n_results=RAG_TOP_K,
                where={"source": filename}
            )

        else:

            results = documents_collection.query(
                query_texts=[question],
                n_results=RAG_TOP_K
            )

        docs = results.get(
            "documents",
            [[]]
        )

        if not docs or not docs[0]:

            return {
                "answer": (
                    "No relevant document content "
                    "was found."
                ),
                "sources": []
            }

        selected_docs = docs[0]

        context = "\n\n".join(
            selected_docs
        )

        context = limit_context(
            context,
            3000
        )

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
Answer using ONLY the document context.

If the answer is not present, say:
"I could not find this information in the uploaded document."

Be short and clear.

DOCUMENT:
{context}

QUESTION:
{question}

ANSWER:
"""

        answer = ask_ollama(prompt)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = []

        if filename:

            sources.append(filename)

        else:

            try:

                metadatas = results.get(
                    "metadatas",
                    [[]]
                )

                if metadatas and metadatas[0]:

                    for metadata in metadatas[0]:

                        source = metadata.get(
                            "source"
                        )

                        if (
                            source
                            and source not in sources
                        ):

                            sources.append(source)

            except Exception:

                pass

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        logger.exception(
            "Question answering failed."
        )

        return {
            "answer": (
                f"⚠️ Could not answer "
                f"the question: {e}"
            ),
            "sources": []
        }


# ============================================================
# GET DOCUMENT CONTENT
# ============================================================

def get_document_content(
    filename: str,
    max_chunks: int = 4,
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    Get limited indexed content for AI processing.
    """

    try:

        results = documents_collection.get(
            where={"source": filename}
        )

        docs = results.get(
            "documents",
            []
        )

        if not docs:

            return ""

        # Only use a small number of chunks
        selected_docs = docs[:max_chunks]

        context = "\n\n".join(
            selected_docs
        )

        return limit_context(
            context,
            max_chars
        )

    except Exception:

        logger.exception(
            "Could not get document content."
        )

        return ""


# ============================================================
# SUMMARY
# ============================================================

def summarize_document(
    filename: str
) -> dict:
    """
    Generate a short document summary.
    """

    filename = filename.strip()

    if not filename:

        return {
            "summary": "Filename cannot be empty.",
            "filename": filename,
            "success": False
        }

    context = get_document_content(
        filename,
        max_chunks=3,
        max_chars=2800
    )

    if not context:

        return {
            "summary": (
                "No indexed content was found "
                "for this item."
            ),
            "filename": filename,
            "success": False
        }

    # --------------------------------------------------------
    # FAST SUMMARY PROMPT
    # --------------------------------------------------------

    prompt = f"""
Summarize this document.

Give:
• Main topic
• Important points
• Conclusion

Use ONLY the content provided.
Do not invent information.

Keep it very short, under 80 words.

CONTENT:
{context}

SUMMARY:
"""

    summary = ask_ollama(prompt)

    if summary.startswith("⚠️"):

        return {
            "summary": summary,
            "filename": filename,
            "success": False
        }

    return {
        "summary": summary,
        "filename": filename,
        "success": True
    }


# ============================================================
# AI SUGGESTIONS
# ============================================================

def generate_suggestions(
    filename: str
) -> dict:
    """
    Generate practical AI suggestions
    from a document or meeting transcript.
    """

    filename = filename.strip()

    if not filename:

        return {
            "suggestions": (
                "Filename cannot be empty."
            ),
            "filename": filename,
            "success": False
        }

    context = get_document_content(
        filename,
        max_chunks=3,
        max_chars=3000
    )

    if not context:

        return {
            "suggestions": (
                "No indexed content was found "
                "for this item."
            ),
            "filename": filename,
            "success": False
        }

    # --------------------------------------------------------
    # FAST SUGGESTION PROMPT
    # --------------------------------------------------------

    prompt = f"""
Analyze the document or meeting transcript.

Give practical suggestions based ONLY
on the provided content.

Format:

Next steps:
- ...

Follow-up:
- ...

Risks or gaps:
- ...

Do not invent information.
Keep it under 90 words.

CONTENT:
{context}

AI SUGGESTIONS:
"""

    suggestions = ask_ollama(prompt)

    if suggestions.startswith("⚠️"):

        return {
            "suggestions": suggestions,
            "filename": filename,
            "success": False
        }

    return {
        "suggestions": suggestions,
        "filename": filename,
        "success": True
    }


# ============================================================
# TEXT SUMMARY
# ============================================================

def summarize_text(
    text: str,
    filename: str = "meeting"
) -> dict:
    """
    Summarize directly supplied meeting/transcript text.
    """

    text = text.strip()

    if not text:

        return {
            "summary": "No text was provided.",
            "filename": filename,
            "success": False
        }

    # Keep transcript context smaller
    context = limit_context(
        text,
        3000
    )

    # --------------------------------------------------------
    # FAST MEETING SUMMARY
    # --------------------------------------------------------

    prompt = f"""
Summarize this meeting transcript.

Include:
• Main discussion
• Decisions
• Action items
• Conclusion

Use ONLY the transcript.
Do not invent information.

Keep it under 90 words.

TRANSCRIPT:
{context}

SUMMARY:
"""

    summary = ask_ollama(prompt)

    if summary.startswith("⚠️"):

        return {
            "summary": summary,
            "filename": filename,
            "success": False
        }

    return {
        "summary": summary,
        "filename": filename,
        "success": True
    }