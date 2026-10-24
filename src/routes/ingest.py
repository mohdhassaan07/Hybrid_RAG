from pathlib import Path
import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile
from src.ingestion import ingest
router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    # 1. Validate that the file is a PDF
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed.",
        )
    # 2. Save the uploaded file to disk
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        # 3. Call your ingestion function with the file path
        ingest(str(file_path))
        return {
            "message": "Data ingested successfully",
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
