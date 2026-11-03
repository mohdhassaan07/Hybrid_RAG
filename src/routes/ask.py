from fastapi import APIRouter
from src.retriever import ask_query
router = APIRouter()

@router.post("/ask")
def ask(query: str):
    return ask_query(query)
    
