import os
import time
from pathlib import Path

from qdrant_client import QdrantClient

from src.ingestion import ingest  # Change to your actual ingestion module


# Configuration
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

SAMPLE_DOCS_DIR = Path("sample_docs")

MAX_RETRIES = 30
RETRY_DELAY = 2


def wait_for_qdrant():
    """Wait until Qdrant is available."""

    print("Waiting for Qdrant to become available...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = QdrantClient(url=QDRANT_URL)
            client.get_collections()

            print("Qdrant is ready!")
            return client

        except Exception as e:
            print(
                f"Attempt {attempt}/{MAX_RETRIES}: "
                f"Qdrant is not ready yet. {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        "Could not connect to Qdrant after maximum retries."
    )


def seed_documents():
    """Index sample documentation into Qdrant and BM25."""

    if not SAMPLE_DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Sample documentation directory not found: "
            f"{SAMPLE_DOCS_DIR.resolve()}"
        )

    supported_extensions = {".pdf", ".txt"}

    sample_files = sorted(
        file
        for file in SAMPLE_DOCS_DIR.rglob("*")
        if file.is_file()
        and file.suffix.lower() in supported_extensions
    )

    if not sample_files:
        raise RuntimeError(
            "No supported sample documents found."
        )

    print(f"\nFound {len(sample_files)} sample documents.")

    total_indexed = 0

    for file in sample_files:
        print(f"\nProcessing: {file.name}")

        try:
            result = ingest(str(file))

            if isinstance(result, dict):
                indexed = result.get("indexed", 0)
                total_indexed += indexed

                print(f"Indexed chunks: {indexed}")
                print(result.get("message", ""))

        except Exception as e:
            raise RuntimeError(
                f"Failed to index {file.name}: {e}"
            ) from e

    print("\n" + "=" * 50)
    print("SEEDING COMPLETED SUCCESSFULLY")
    print(f"Documents processed: {len(sample_files)}")
    print(f"New chunks indexed: {total_indexed}")
    print("=" * 50)


if __name__ == "__main__":

    wait_for_qdrant()

    seed_documents()
