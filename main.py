import os

# Настройка кэша
PATH = './hf_cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Локальные функции RAG
from retriever import retrieve_chunks
from generator import generate_answer

# Модели эмбеддинга и база данных
from sentence_transformers import SentenceTransformer
import chromadb

# Модели LLM (Mistral/Fallback)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



# --- Глобальная инициализация ---
embed_model = None
chroma_client = None
collection = None
llm_tokenizer = None
llm_model = None
device = None

# def get_device():
#     return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Модели запросов и ответов
class QueryResponse(BaseModel):
    query: str
    top_k: int
    chunks: List[dict]
    answer: str

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
    print("Starting...")
    """Инициализируем эмбеддинги, ChromaDB и LLM на нужном устройстве"""
    global embed_model, chroma_client, collection, llm_tokenizer, llm_model, device
    load_dotenv()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Environment variables loaded")

    # 1. Модель эмбеддингов
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large')

    print("Embedding model loaded")

    # 2. ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chromadb_data")
    collection = chroma_client.get_collection(name="yandex_foundation_models_docs")

    print("Chromadb collection loaded")

    # 3. LLM и устройство

    print("Device loaded: {}".format(device))
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    auth_kwargs = {'use_auth_token': hf_token} if hf_token else {}
    preferred = "mistralai/Mistral-7B-Instruct-v0.3"
    fallback = "tiiuae/falcon-7b-instruct"
    print("Preferred: {}".format(preferred))
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(preferred, **auth_kwargs)
        print("Tokenizer loaded")
        llm_model = AutoModelForCausalLM.from_pretrained(
            preferred,
            torch_dtype=torch.float16 if device.type=='cuda' else torch.float32,
            **auth_kwargs
        )
        print("LLM Model loaded")
    except Exception as e:
        print("LLM Model could not be loaded")
        print(e)
        llm_tokenizer = AutoTokenizer.from_pretrained(fallback)
        llm_model = AutoModelForCausalLM.from_pretrained(fallback)
    llm_model.to(device)
    print("LLM Model loaded to {}".format(device.type))

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(query: str, top_k: int = 5):
    """
    top_k — количество релевантных чанков для ретривала.
    Генерация идёт на устройстве: GPU, если доступна.
    """
    try:
        print('1.0')
        results = retrieve_chunks(query, top_k, embed_model, collection)
        print('1.1')
        chunks = [
            {"text": doc, "source": meta['source'], "distance": dist}
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
        print('1.2')
        answer = generate_answer(query, chunks, llm_model, llm_tokenizer, device)
        print('1.3')
        return QueryResponse(query=query, top_k=top_k, chunks=chunks, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload", response_model=ReloadResponse, tags=["Admin"])
async def reload_index():
    try:
        from indexer import reindex
        reindex()
        return ReloadResponse(status="success", message="Index reloaded")
    except Exception as e:
        return ReloadResponse(status="failure", message=str(e))

@app.get("/admin/status", response_model=ReloadResponse, tags=["Admin"])
async def index_status():
    status = "ok" if collection else "error"
    msg = "Index is ready" if collection else "Index not initialized"
    return ReloadResponse(status=status, message=msg)

# Запуск:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
