from fastapi import FastAPI
from src.routes.ask import router as ask_router
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(ask_router,prefix="/v1")
