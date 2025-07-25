import os
from dotenv import load_dotenv
PATH = './hf_cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional


# Локальные функции RAG
from retriever import retrieve_chunks
from generator import generate_answer

# Модели эмбеддинга и база данных
from sentence_transformers import SentenceTransformer
import chromadb

# Модели LLM (Mistral)
from transformers import AutoModelForCausalLM, AutoTokenizer


print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("HF_HOME:", os.environ.get("HF_HOME"))



# --- Глобальная инициализация ---
# Embedding
embed_model = None
# ChromaDB
chroma_client = None
collection = None
# LLM
llm_tokenizer = None
llm_model = None

# Модели запросов и ответов
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Chunk(BaseModel):
    text: str
    source: str
    distance: float

class QueryResponse(BaseModel):
    query: str
    chunks: List[Chunk]
    answer: str

# Административные модели
class ReloadResponse(BaseModel):
    status: str
    message: Optional[str] = None

# Инициализация FastAPI
app = FastAPI(
    title="RAG System for Yandex Foundation Models",
    version="0.1.0",
    description="Retrieval-Augmented Generation API"
)

@app.on_event("startup")
async def startup_event():
    """Инициализируем эмбеддинги, ChromaDB и LLM"""
    print("Starting...")

    global embed_model, chroma_client, collection, llm_tokenizer, llm_model
    load_dotenv()
    # 1. Модель эмбеддингов
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large')

    # 2. ChromaDB клиент и коллекция
    chroma_client = chromadb.PersistentClient(path="./chromadb_data")
    collection = chroma_client.get_collection(name="yandex_foundation_models_docs")

    # 3. LLM Mistral
    # Используем токен из переменной окружения, если необходимо
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    auth_kwargs = {'use_auth_token': hf_token} if hf_token else {}
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            **auth_kwargs
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            torch_dtype="auto",
            **auth_kwargs
        )
    except Exception as e:
        # Если модель недоступна, используем fallback
        print(f"Warning: failed to load gated Mistral model: {e}")
        # Пример публичной альтернативы
        fallback_model = "tiiuae/falcon-7b-instruct"
        llm_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        llm_model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            device_map="auto",
            torch_dtype="auto"
        )
        print(f"Loaded fallback model: {fallback_model}")

    print("Started!")

@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка состояния сервиса"""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """Обрабатывает запрос: ретрив и генерация ответа"""
    try:
        # 1. Ретрив
        results = retrieve_chunks(
            request.query,
            request.top_k,
            embed_model,
            collection
        )
        chunks = [Chunk(text=doc, source=meta['source'], distance=dist)
                  for doc, meta, dist in zip(
                      results['documents'][0],
                      results['metadatas'][0],
                      results['distances'][0]
                  )]

        # 2. Генерация ответа
        answer = generate_answer(
            query=request.query,
            chunks=chunks,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer
        )

        return QueryResponse(
            query=request.query,
            chunks=chunks,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload", response_model=ReloadResponse, tags=["Admin"])
async def reload_index():
    """Перезагружает индекс: повторно выполняет индексирование"""
    try:
        from indexer import reindex
        reindex()
        return ReloadResponse(status="success", message="Index reloaded")
    except Exception as e:
        return ReloadResponse(status="failure", message=str(e))

@app.get("/admin/status", response_model=ReloadResponse, tags=["Admin"])
async def index_status():
    """Возвращает состояние индекса и модели"""
    status = "ok" if collection is not None else "error"
    msg = "Index is ready" if collection is not None else "Index not initialized"
    return ReloadResponse(status=status, message=msg)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
