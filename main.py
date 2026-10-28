from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
import logging
from datetime import datetime

import rag

from faster_whisper import WhisperModel


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Meeting & Document Intelligence System"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_FOLDER = "../documents"
TRANSCRIPTS_FOLDER = os.path.join(DOCUMENTS_FOLDER, "transcripts")

ALLOWED_DOC_TYPES = {
    ".pdf",
    ".docx",
    ".txt"
}

ALLOWED_AUDIO_TYPES = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".webm",
    ".ogg"
}

MAX_FILE_SIZE_MB = 200


os.makedirs(
    DOCUMENTS_FOLDER,
    exist_ok=True
)

os.makedirs(
    TRANSCRIPTS_FOLDER,
    exist_ok=True
)


# ============================================================
# SAFE FILE PATH
# ============================================================

def safe_save_path(
    filename: str
) -> tuple[str, str]:

    ext = os.path.splitext(
        filename
    )[1].lower()

    stored_filename = (
        f"{uuid.uuid4().hex}{ext}"
    )

    full_path = os.path.join(
        DOCUMENTS_FOLDER,
        stored_filename
    )

    return stored_filename, full_path


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_upload(
    file: UploadFile,
    allowed_types: set[str]
) -> tuple[str, str]:

    original_filename = (
        file.filename or ""
    )

    ext = os.path.splitext(
        original_filename
    )[1].lower()


    if ext not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(allowed_types)}"
            )
        )


    content = await file.read()


    size_mb = (
        len(content) /
        (1024 * 1024)
    )


    if size_mb > MAX_FILE_SIZE_MB:

        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large "
                f"({size_mb:.1f} MB). "
                f"Maximum is "
                f"{MAX_FILE_SIZE_MB} MB."
            )
        )


    if size_mb == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )


    stored_filename, full_path = (
        safe_save_path(original_filename)
    )


    try:

        with open(
            full_path,
            "wb"
        ) as buffer:

            buffer.write(content)

    except OSError as e:

        logger.exception(
            "Failed to save uploaded file"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {e}"
        )


    return (
        original_filename,
        full_path
    )


# ============================================================
# WHISPER MODEL
# Lazy loaded once
# ============================================================

whisper_model = None


def get_whisper_model() -> WhisperModel:

    global whisper_model


    if whisper_model is None:

        logger.info(
            "Loading Whisper tiny model..."
        )

        whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )


    return whisper_model


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(
    audio_path: str
) -> str:

    model = get_whisper_model()


    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        best_of=5,
        vad_filter=True,

        condition_on_previous_text=True
    )


    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    )


    return transcript.strip()


# ============================================================
# SAVE TRANSCRIPT WITH DATE/TIME
# ============================================================

def save_transcript_with_timestamp(
    transcript: str,
    original_name: str | None = None
) -> str:

    meeting_label = (
        os.path.splitext(
            os.path.basename(
                original_name or "meeting"
            )
        )[0]
        or "meeting"
    )

    timestamp = (
        datetime.now()
        .strftime("%Y-%m-%d_%H-%M-%S")
    )

    transcript_filename = (
        f"{meeting_label}_{timestamp}.txt"
    )

    transcript_path = os.path.join(
        TRANSCRIPTS_FOLDER,
        transcript_filename
    )

    with open(
        transcript_path,
        "w",
        encoding="utf-8"
    ) as transcript_file:
        transcript_file.write(transcript)

    return transcript_filename


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message":
        "AI Meeting & Document Intelligence System is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...)
):

    original_name, file_path = (
        await save_upload(
            file,
            ALLOWED_DOC_TYPES
        )
    )


    try:

        logger.info(
            "Extracting text from %s",
            original_name
        )


        text = rag.extract_text(
            file_path
        )


        if not text or not text.strip():

            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text "
                    "found in this document."
                )
            )


        chunks_stored = (
            rag.store_document_chunks(
                original_name,
                text
            )
        )


    except ValueError as e:

        logger.exception(
            "Document indexing failed"
        )

        return {

            "message":
            f"File saved, but not indexed: {e}",

            "filename":
            original_name,

            "success":
            False
        }


    except HTTPException:

        raise


    except Exception as e:

        logger.exception(
            "Indexing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {e}"
        )


    return {

        "message":
        "Document uploaded and indexed successfully",

        "filename":
        original_name,

        "chunks_stored":
        chunks_stored,

        "success":
        True
    }


# ============================================================
# ASK AI
# ============================================================

@app.post("/ask")
async def ask(

    question: str = Form(...),

    filename: str | None = Form(None)

):

    question = question.strip()


    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # If filename is supplied by frontend,
    # RAG searches ONLY that document.
    # --------------------------------------------------------

    if filename:

        filename = filename.strip()


    logger.info(
        "Question received: %s",
        question
    )

    logger.info(
        "Active document: %s",
        filename
    )


    try:

        result = rag.answer_question(
            question,
            filename
        )


    except Exception as e:

        logger.exception(
            "Question answering failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not answer question: {e}"
            )
        )


    return result
# ============================================================
# SUMMARY
# ============================================================

@app.post("/summary")
async def summary(
    filename: str = Form(...)
):

    filename = filename.strip()

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty."
        )

    logger.info(
        "Generating summary for: %s",
        filename
    )

    try:

        result = rag.summarize_document(
            filename
        )

    except Exception as e:

        logger.exception(
            "Summarization failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {e}"
        )

    return result


# ============================================================
# AI SUGGESTIONS
# ============================================================

@app.post("/suggestions")
async def suggestions(
    filename: str = Form(...)
):

    filename = filename.strip()

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty."
        )

    logger.info(
        "Generating AI suggestions for: %s",
        filename
    )

    try:

        return rag.generate_suggestions(
            filename
        )

    except Exception as e:

        logger.exception(
            "Suggestion generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Suggestion generation failed: {e}"
        )


# ============================================================
# MEETING SUMMARY
# ============================================================

@app.post("/meeting-summary")
async def meeting_summary(
    filename: str = Form(...)
):

    filename = filename.strip()

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty."
        )

    try:

        results = rag.documents_collection.get(
            where={
                "source": filename
            }
        )

        docs = results.get(
            "documents",
            []
        )

        if not docs:

            return {
                "summary":
                    "No meeting transcript found.",

                "filename":
                    filename,

                "success":
                    False
            }

        transcript = "\n\n".join(
            docs
        )

        return rag.summarize_text(
            transcript,
            filename
        )

    except Exception as e:

        logger.exception(
            "Meeting summarization failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Meeting summarization failed: {e}"
            )
        )



# ============================================================
# TRANSCRIBE MEETING
# ============================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):

    original_name, audio_path = (
        await save_upload(
            file,
            ALLOWED_AUDIO_TYPES
        )
    )


    logger.info(
        "Transcribing meeting: %s",
        original_name
    )


    try:

        transcript = transcribe_audio(
            audio_path
        )


        if not transcript:

            return {

                "message":
                "No speech was detected in the recording.",

                "filename":
                original_name,

                "transcript":
                "",

                "chunks_stored":
                0,

                "success":
                False
            }


        # ----------------------------------------------------
        # IMPORTANT:
        # Store transcript in ChromaDB.
        #
        # This allows:
        # Meeting → Transcript → RAG → Ask AI / Summary
        # ----------------------------------------------------

        chunks_stored = (
            rag.store_document_chunks(
                original_name,
                transcript
            )
        )

        saved_transcript = (
            save_transcript_with_timestamp(
                transcript,
                original_name
            )
        )


    except Exception as e:

        logger.exception(
            "Transcription failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Transcription failed: {e}"
            )
        )


    return {

        "message":
        "Audio transcribed and indexed successfully",

        "filename":
        original_name,

        "transcript":
        transcript,

        "saved_transcript":
        saved_transcript,

        "chunks_stored":
        chunks_stored,

        "success":
        True
    }