# src/rag/api.py (VERSÃO FINAL E CORRIGIDA)
import os # Importe o módulo 'os'
import pickle
import faiss
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# --- Modelos Pydantic ---
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class Source(BaseModel):
    page: int
    snippet: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

# --- Variáveis Globais e Configuração ---
DB_PATH = "db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# CORREÇÃO DEFINITIVA:
# Lê a URL do Ollama da variável de ambiente. Se não for encontrada,
# usa 'localhost' como padrão para desenvolvimento local.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carregar recursos na inicialização
    print("Loading RAG model and data...")
    ml_models["embedding_model"] = SentenceTransformer(MODEL_NAME)
    ml_models["index"] = faiss.read_index(f"{DB_PATH}/vector_index.faiss")
    with open(f"{DB_PATH}/chunks.pkl", "rb") as f:
        ml_models["chunks"] = pickle.load(f)
    print("Resources loaded successfully.")
    yield
    # Limpar recursos no desligamento
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# ... (o resto do arquivo continua exatamente igual) ...

def retrieve_context(question: str, top_k: int) -> list[dict]:
    model = ml_models["embedding_model"]
    index = ml_models["index"]
    chunks = ml_models["chunks"]
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_answer(question: str, context: list[dict]) -> str:
    context_str = "\n---\n".join([f"Fonte (Página {c['page']}):\n{c['snippet']}" for c in context])
    prompt = f"""
    Você é um assistente técnico especialista. Baseado ESTREITAMENTE no contexto fornecido abaixo, responda a pergunta do usuário.
    Se a resposta não estiver no contexto, responda: "Não encontrei informações sobre isso nas especificações técnicas."
    Cite sempre a página da fonte.

    **Contexto:**
    {context_str}

    **Pergunta:**
    {question}

    **Resposta:**
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "llama3.1", "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        raise HTTPException(status_code=500, detail="Failed to communicate with the language model.")

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not all(k in ml_models for k in ["embedding_model", "index", "chunks"]):
        raise HTTPException(status_code=503, detail="Model and data not loaded yet. Please wait.")
    context_chunks = retrieve_context(request.question, request.top_k)
    if not context_chunks:
        return AskResponse(
            answer="Não foi possível encontrar trechos relevantes no documento para responder a sua pergunta.",
            sources=[]
        )
    answer = generate_answer(request.question, context_chunks)
    sources = [Source(page=chunk["page"], snippet=chunk["snippet"]) for chunk in context_chunks]
    return AskResponse(answer=answer, sources=sources)

@app.get("/health")
def health_check():
    return {"status": "ok"}