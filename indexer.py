import os
import git
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
PATH = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

# 1. Клонирование репозитория
REPO_URL = "https://github.com/yandex-cloud/docs.git"
REPO_DIR = "./yandex_docs"
FOUNDATION_MODELS_PATH = os.path.join(REPO_DIR, "ru", "foundation-models")

if not os.path.exists(REPO_DIR):
    print(f"Клонирование репозитория {REPO_URL} в {REPO_DIR}...")
    git.Repo.clone_from(REPO_URL, REPO_DIR)
    print("Репозиторий успешно склонирован.")
else:
    print(f"Репозиторий уже существует в {REPO_DIR}. Пропускаем клонирование.")

# 2. Чтение и обработка Markdown файлов
print(f"Чтение Markdown файлов из {FOUNDATION_MODELS_PATH}...")
documents = []
for root, _, files in os.walk(FOUNDATION_MODELS_PATH):
    for file in files:
        if file.endswith(".md"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Очистка HTML-тегов, если они есть в Markdown
                soup = BeautifulSoup(content, 'html.parser')
                clean_content = soup.get_text()
                documents.append({"content": clean_content, "source": filepath})
print(f"Найдено {len(documents)} документов.")

# 3. Разбиение текста на чанки
print("Разбиение документов на чанки...")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = []
for doc in documents:
    doc_chunks = text_splitter.split_text(doc["content"])
    for i, chunk in enumerate(doc_chunks):
        chunks.append({
            "text": chunk,
            "source": doc["source"],
            "chunk_id": f"{doc['source']}_{i}"
        })
print(f"Создано {len(chunks)} чанков.")

# 4. Векторизация чанков
print("Загрузка модели векторизации (intfloat/multilingual-e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

print("Векторизация чанков...")
chunk_texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(chunk_texts, show_progress_bar=True)

for i, embedding in enumerate(embeddings):
    chunks[i]["embedding"] = embedding.tolist() # Сохраняем как список для ChromaDB
print("Векторизация завершена.")

# 5. Сохранение в ChromaDB
print("Инициализация ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb_data")
collection_name = "yandex_foundation_models_docs"

# Удаляем коллекцию, если она уже существует, для чистого запуска
try:
    client.delete_collection(name=collection_name)
    print(f"Существующая коллекция '{collection_name}' удалена.")
except:
    pass # Коллекция не существует, это нормально

collection = client.create_collection(name=collection_name)

print("Добавление чанков в ChromaDB...")
ids = [chunk["chunk_id"] for chunk in chunks]
metadatas = [{
    "source": chunk["source"],
    "text_length": len(chunk["text"])
} for chunk in chunks]

# ChromaDB ожидает список списков для embeddings
embeddings_list = [chunk["embedding"] for chunk in chunks]

collection.add(
    embeddings=embeddings_list,
    documents=chunk_texts,
    metadatas=metadatas,
    ids=ids
)
print(f"Все {len(chunks)} чанков успешно добавлены в ChromaDB.")
print("Индексация завершена. Данные сохранены в ./chromadb_data")

