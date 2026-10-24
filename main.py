from fastapi import FastAPI
from src.routes.ask import router as ask_router
from src.routes.ingest import router as ingest_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(ask_router,prefix="/v1")
app.include_router(ingest_router,prefix="/v1")
