import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MODEL_NAME = "gpt-4.1-mini"
# 1. Инициализация модели эмбеддингов (общая для всех тестов)
embed_model = SentenceTransformer('intfloat/multilingual-e5-large')

def get_balanced_chunks(text, chunk_size):
    """
    Делит текст на N равных частей, чтобы каждая была <= max_chunk_size.
    """
    print(f"Делим на {chunk_size} чанка")
    
    # Настраиваем сплиттер на этот идеальный размер
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=MODEL_NAME,
        chunk_size=chunk_size+20, # Небольшой запас для гибкости разделителей
        chunk_overlap=10,
        separators=["\n\n", "\n", "###", ".", " ", ""]
    )
    
    return splitter.split_text(text)

def run_rag_needle_test(full_context, token_size, question_data):
    """
    RAG для ОДНОГО вопроса (question_data — это Dict с ключом 'q')
    """
    chunk_size = 100
    # ШАГ 1: Нарезка
    chunks = get_balanced_chunks(full_context, chunk_size)
    formatted_chunks = ["passage: " + c for c in chunks]
    
    # ШАГ 2: Индексация документов
    embeddings = embed_model.encode(
        formatted_chunks,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    query_text = "query: " + question_data['q']
    query_vec = embed_model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    distances, indices = index.search(query_vec.astype('float32'), k=token_size//100)
    
    # Извлекаем индексы из первой (и единственной) строки результата поиска
    question_indices = indices[0]
    
    # ШАГ 4: Формирование контекста
    block_header = f"=== КОНТЕКСТ ДЛЯ ВОПРОСА: {question_data['q']} ==="
    
    retrieved_chunks = []
    for idx in question_indices:
        if idx != -1 and idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
    
    chunks_text = "\n---\n".join(retrieved_chunks)
    
    # Возвращаем готовую строку контекста
    return f"{block_header}\n{chunks_text}"