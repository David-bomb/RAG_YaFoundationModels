import os

# Настройка папки установки моделей с HF
PATH = './hf_cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

# Локальные функции RAG
from retriever import retrieve_chunks
from generator import generate_answer

from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Глобальная инициализация ---
embed_model = None
chroma_client = None
collection = None
llm_tokenizer = None
llm_model = None
device = None

loaded_models_info = {
    "embedding_model": None,
    "llm_model": None,
    "quantization": None,
    "fallback_used": False
}


# Модели запросов и ответов
class QueryResponse(BaseModel):
    query: str
    top_k: int
    chunks: List[dict]
    answer: str

class StatusResponse(BaseModel):
    status: str
    database_status: str
    llm_status: str
    details: dict

app = FastAPI(
    title="RAG System for Yandex Foundation Models",
    version="0.1.0",
    description="Retrieval-Augmented Generation API"
)

@app.on_event("startup")
async def startup_event():
    """Инициализируем эмбеддинги, ChromaDB и LLM на нужном устройстве"""

    print("Starting...")
    global embed_model, chroma_client, collection, llm_tokenizer, llm_model, device
    load_dotenv()
    print("Environment variables loaded")

    embed_model_name = 'intfloat/multilingual-e5-large'
    embed_model = SentenceTransformer(embed_model_name)
    loaded_models_info["embedding_model"] = embed_model_name
    print("Embedding model loaded")

    chroma_client = chromadb.PersistentClient(path="./chromadb_data")
    collection = chroma_client.get_collection(name="yandex_foundation_models_docs")
    print("Chromadb collection loaded")

    hf_token = os.getenv('HUGGINGFACE_TOKEN') # Для этого должен быть .env файл с вашим токеном
    auth_kwargs = {'use_auth_token': hf_token} if hf_token else {}
    preferred = "mistralai/Mistral-7B-Instruct-v0.3"
    fallback = "tiiuae/falcon-7b-instruct"
    print("Preferred: {}".format(preferred))

    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(preferred, **auth_kwargs)
        print("Tokenizer loaded")
        llm_model = AutoModelForCausalLM.from_pretrained(
            preferred,
            load_in_8bit=True, # или load_in_4bit=True для максимальной экономии
            device_map="auto",
            **auth_kwargs
        )
        print("LLM loaded")
        loaded_models_info["llm_model"] = preferred
        loaded_models_info["quantization"] = "8-bit"
        loaded_models_info["fallback_used"] = False

    except Exception as e:
        print("LLM could not be loaded")
        print(e)
        llm_tokenizer = AutoTokenizer.from_pretrained(fallback)
        llm_model = AutoModelForCausalLM.from_pretrained(fallback)
        loaded_models_info["llm_model"] = fallback
        loaded_models_info["quantization"] = "None"
        loaded_models_info["fallback_used"] = True

    print("Startup is over.")


@app.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(query: str, top_k: int = 5):
    """
    top_k — количество релевантных чанков для ретривала.
    Генерация идёт на устройстве: GPU, если доступна.
    """
    try:
        print('Query accepted...')
        results = retrieve_chunks(query, top_k, embed_model, collection)
        chunks = [
            {"text": doc, "source": meta['source'], "distance": dist}
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
        print('Chunks found...')
        answer = generate_answer(query, chunks, llm_model, llm_tokenizer)
        print('Answer generated!')
        return QueryResponse(query=query, top_k=top_k, chunks=chunks, answer=answer)
    except Exception as e:
        print('Query failed.')
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_system_status():
    """
    Проверяет состояние системы, включая базу данных и LLM.
    Возвращает детальную информацию о загруженных моделях.
    """
    db_status = "ok"
    llm_status = "ok"
    overall_status = "ok"

    # 1. Проверка состояния ChromaDB
    try:
        collection.peek(limit=1)
        print("Database check successful.")
    except Exception as e:
        db_status = "error"
        overall_status = "degraded"
        print(f"Database check failed: {e}")

    # 2. Проверка состояния LLM
    if not llm_model or not llm_tokenizer:
        llm_status = "error"
        overall_status = "degraded"
        print("LLM check failed: model or tokenizer not loaded.")
    else:
        print("LLM check successful.")

    return StatusResponse(
        status=overall_status,
        database_status=db_status,
        llm_status=llm_status,
        details=loaded_models_info
    )

@app.get("/", response_class=FileResponse)
async def read_root():
    return 'static/index.html'

# Запуск:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
