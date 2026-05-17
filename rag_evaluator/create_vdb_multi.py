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
        chunk_size=chunk_size+50, # Небольшой запас для гибкости разделителей
        chunk_overlap=25,
        separators=["\n\n", "\n", "###", ".", " ", ""]
    )
    
    return splitter.split_text(text)

def run_rag_multi_needle_test(full_context, chunk_size, questions_data):
    chunks = get_balanced_chunks(full_context, chunk_size)
    formatted_chunks = ["passage: " + c for c in chunks]
    
    embeddings = embed_model.encode(formatted_chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    query_texts = ["query: " + item['q'] for item in questions_data]
    query_vecs = embed_model.encode(query_texts, batch_size=len(questions_data), convert_to_numpy=True)
    
    _, indices = index.search(query_vecs.astype('float32'), k=5)
    
    final_results = []
    
    for i in range(len(questions_data)):
        question_indices = indices[i]
        
        # Заголовок для группы чанков конкретного вопроса
        block_header = f"=== КОНТЕКСТ ДЛЯ ВОПРОСА №{i+1}: {questions_data[i]['q']} ==="
        
        retrieved_chunks = []
        for idx in question_indices:
            if idx != -1 and idx < len(chunks):
                retrieved_chunks.append(chunks[idx])
        
        # Склеиваем только сами тексты чанков
        chunks_text = "\n".join(retrieved_chunks)
        
        # Формируем цельный блок для этого вопроса
        full_block = f"{block_header}\n{chunks_text}"
        final_results.append(full_block)

    return final_results