import os

# Настройка кэша
PATH = './hf_cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import git
import re
# Импортируем MarkdownHeaderTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# --- Конфигурация ---
REPO_URL = "https://github.com/yandex-cloud/docs.git"
REPO_DIR = "./yandex_docs"
FOUNDATION_MODELS_PATH = os.path.join(REPO_DIR, "ru", "foundation-models")
CHROMA_DB_PATH = "./chromadb_data"
COLLECTION_NAME = "yandex_foundation_models_docs"
BATCH_SIZE = 32


def clean_markdown_content(content: str) -> str:
    """
    Очищает Markdown-контент от специфичных конструкций и синтаксиса.
    """
    # Удаление блоков {% ... %} и инлайновых {{ ... }}
    content = re.sub(r'\{%.*?%\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{.*?\}\}', '', content)

    # ИЗМЕНЕНО: Добавлено правило для удаления Markdown-ссылок.
    # Заменяет [текст](ссылка) на просто "текст".
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

    return content


# 1. Клонирование репозитория
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
                if content.strip():
                    documents.append({"content": content, "source": filepath})
print(f"Найдено {len(documents)} документов.")

# 3. Разбиение текста на чанки
print("Разбиение документов на чанки с помощью MarkdownHeaderTextSplitter...")
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = []
for doc in documents:
    # Сначала очищаем контент от ВСЕХ видов мусора
    clean_content = clean_markdown_content(doc["content"])
    # Затем разбиваем на фрагменты по заголовкам
    doc_fragments = markdown_splitter.split_text(clean_content)
    for i, fragment in enumerate(doc_fragments):
        chunks.append({
            "text": fragment.page_content,
            "source": doc["source"],
            "metadata": fragment.metadata,
            "chunk_id": f"{doc['source']}_{i}"
        })
print(f"Создано {len(chunks)} чанков.")

# 4. Векторизация чанков
print("Загрузка модели векторизации (intfloat/multilingual-e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

# 5. Сохранение в ChromaDB
print("Инициализация ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Существующая коллекция '{COLLECTION_NAME}' удалена.")
except Exception:
    pass

collection = client.create_collection(name=COLLECTION_NAME)

print(f"Векторизация и добавление чанков в ChromaDB пакетами по {BATCH_SIZE}...")

total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Обработка батчей"):
    batch_chunks = chunks[i:i + BATCH_SIZE]

    batch_texts = [chunk["text"] for chunk in batch_chunks]
    batch_ids = [chunk["chunk_id"] for chunk in batch_chunks]
    batch_metadatas = [{"source": chunk["source"]} for chunk in batch_chunks]

    batch_embeddings = model.encode(batch_texts)

    collection.add(
        embeddings=batch_embeddings.tolist(),
        documents=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

print(f"Все {len(chunks)} чанков успешно добавлены в ChromaDB.")
print(f"Индексация завершена. Данные сохранены в {CHROMA_DB_PATH}")
