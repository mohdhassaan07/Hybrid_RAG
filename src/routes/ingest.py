from pathlib import Path
import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from src.ingestion import ingest, ingest_text

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request schema for raw-text ingestion
# ---------------------------------------------------------------------------

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1,
                      description="Raw text content to index.")
    source_label: str = Field(
        default="inline_text",
        description="Label stored as 'source' in chunk metadata (e.g. document title).",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    """Ingest a PDF or .txt file uploaded as multipart/form-data."""
    if file.content_type != "application/pdf" and not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and .txt files are allowed.",
        )
    # Save the uploaded file to disk
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        ingest(str(file_path))
        return {
            "message": "Data ingested successfully",
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@router.post("/ingest/text")
async def ingest_text_data(body: IngestTextRequest):
    """Ingest raw text directly — no file upload needed.

    Send a JSON body with:
    - **text**: the content to index (required)
    - **source_label**: a human-readable name stored in chunk metadata (optional)

    Example:
    ```json
    {
      "text": "Knowledge Graph RAG stores entities as nodes ...",
      "source_label": "my_notes"
    }
    ```
    """
    try:
        result = ingest_text(body.text, source_label=body.source_label)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
