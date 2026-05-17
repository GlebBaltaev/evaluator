from openai import OpenAI
import json
import re
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple, Dict, Any
import traceback
from create_vdb import run_rag_multi_needle_test

ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})




#gpt or deepseek
client = OpenAI(api_key="",
                base_url = "")


with open('../text_needles/text_needles.json', 'r') as file:
    needles = json.load(file)
    
system_estimator = """You are a discrete estimator of correctness for a set of 6 answers. 
You will be given 6 pairs of (Question + Truly Correct Answer) and 6 corresponding (Model Responses).

YOUR TASK:
For each pair, compare the Model Response with the Truly Correct Answer.
- If the meaning of the correct answer is present in the response (even with extra info or interpretation), the score is 1.
- If the response says it doesn't know (when the info was there) or gives a different fact, the score is 0.

RULES:
1. Ignore fluff, polite phrases, and extra information.
2. Focus on the core semantic match.

OUTPUT FORMAT:
Provide your explanation for each question in Russian, and end EACH line with the score [0] or [1]. 

Example:
1. В ответе упомянута экономика и инструменты, смысл совпадает. [1]
2. Модель ответила, что не знает пароль, хотя он был в тексте. [0]
3. Несмотря на избыточность правильный ответ был дан [1]
... and so on."""


# system_answer = """You are a data extraction assistant. Your sole task is to answer questions using ONLY the provided CONTEXT. 

# STRICT RULES:
# 1. If the context does not contain the information needed to answer a question, state: "Information not found."
# 2. Use ONLY the provided context. Do not use external knowledge. 
# 3. If the text contains information that is factually incorrect or unrealistic in real life, ignore the real-world facts and answer exactly as written in the text.
# 4. Provide the answer strictly according to the template below. Do not include introductory remarks, reasoning, or "Based on the text" phrases.

# ANSWER TEMPLATE:
# Question1: [concise answer]
# Question2: [concise answer]
# Question3: [concise answer]
# Question4: [concise answer]
# Question5: [concise answer]
# Question6: [concise answer]"""

system_answer = "Answer the user's question based on the CONTEXT. In answer don't ask something. If you don`t know answer, say that you don`t know answer because text hasn`t got information about this. Do not pay attention to the fact that the answer is not real in life, if it is written in the text, then it is so."

tokens = [8000] #[2000, 4000, 8000, 12000, 14000, 16000, 18000, 20000]
chunk_sizes = [1600] #[50, 100, 200, 400, 450, 500] k=5

MAX_RETRIES = 1
RETRY_DELAY = 5  # секунд
MODEL_NAME = "dolphin3"
MODEL_JUDGE = "deepseek-chat"
DATA_FILE = "data_rag.json"
OUTPUT_FILE = f'../rag_result/{MODEL_NAME}_8k_rag.json'
MAX_WORKERS = 1  # Максимальное количество параллельных запросов
SERVER_URL = "http://localhost:11434/api/chat"


results_dict = {}
existing = set()
existing_lock = threading.Lock()
results_lock = threading.Lock()

# 2. Загрузка из файла (если он есть)
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
            for block in raw_data:
                if isinstance(block, list) and len(block) > 0:
                    # Находим первый тест в блоке для определения ключа (токены, чанк)
                    first_test = next((item for item in block if item.get("block_type") != "statistics_summary"), None)
                    
                    if first_test:
                        t_val = first_test.get("token_size") or first_test.get("num_tokens")
                        c_val = first_test.get("chunk_size")
                        key = (t_val, c_val)
                        
                        if key not in results_dict:
                            results_dict[key] = []
                        
                        # Выделяем только объекты тестов (без статистики)
                        actual_tests = [item for item in block if item.get("block_type") != "statistics_summary"]
                        
                        # Добавляем тесты в основной словарь результатов
                        results_dict[key].extend(actual_tests)
                        
                        # --- ВОТ ЭТОГО НЕ ХВАТАЛО: Наполняем existing ---
                        with existing_lock:
                            for r in actual_tests:
                                # Ключ должен совпадать с тем, что мы проверяем в paralel_processing_multi
                                existing.add((t_val, c_val, r["test_i"]))
                                
        except json.JSONDecodeError:
            print("Ошибка чтения JSON, начинаем с чистого листа.")
            results_dict = {}

print(f"Загружено из файла: {len(existing)} готовых тестов.")

def save_results():
    """Потокобезопасное сохранение с сортировкой блоков и обновлением статистики"""
    final_to_save = []
    
    with results_lock:
        # 1. Сначала сортируем ключи словаря, чтобы блоки в JSON шли по порядку
        # Сортировка сначала по t (токены), затем по p (чанк)
        sorted_keys = sorted(results_dict.keys(), key=lambda x: (x[0], x[1]))
        
        for (t, p) in sorted_keys:
            items = results_dict[(t, p)]
            
            total_needles = len(needles) * 6
            total_needles_in_block = sum(i.get('total_needles', 6) for i in items)
            found_needles_in_block = sum(i.get('found_count', 0) for i in items)            
            processed_tests = len(items)
            
            stats = {
                "block_type": "statistics_summary", 
                "token_size": t, 
                "chunk_size": p, 
                "completed_tests": processed_tests, 
                "total_needles": total_needles,
                "total_needles_tested": total_needles_in_block, 
                "total_needles_found": found_needles_in_block, 
                "success_rate_percent": (found_needles_in_block / total_needles_in_block)
            }
            
            # 2. Сортируем сами тесты внутри блока по их индексу test_i
            # Это удобно, чтобы в файле они шли как 0, 1, 2... 49
            sorted_items = sorted(items, key=lambda x: x.get('test_i', 0))
            
            # Собираем итоговый блок
            block_with_stats = sorted_items + [stats]
            final_to_save.append(block_with_stats)
            
        # 3. Записываем отсортированный список в файл
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_to_save, f, ensure_ascii=False, indent=2)
            
def request_with_retry(func, *args, **kwargs):
    """Выполняет функцию с ретраями"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Ошибка (попытка {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None

def ollama_call(payload):
    """Функция для вызова Ollama"""
    resp = ollama_session.post(SERVER_URL, json=payload, timeout=500)
    resp.raise_for_status()
    time.sleep(10)
    return resp.json()

def process_multi_needle_task(base_content: str, needles_batch: List[Dict], chunk_size: int, token_size: int, test_i: int) -> List[Dict]:
    results = []
    task_key = (token_size, chunk_size, test_i) 
    
    try:
        # 1. Формируем полный текст с вставленными иголками (для RAG)
        full_context_with_needles = base_content
        for item in needles_batch:
            full_context_with_needles = full_context_with_needles.replace('0xFFFF', f"\n{item['inf']}\n", 1)

        # 2. RAG: Получаем список контекстов (по одному на каждый вопрос)
        # Ожидаем, что run_rag_multi_needle_test вернет List[str] длиной 6
        retrieved_contexts_list = run_rag_multi_needle_test(full_context_with_needles, chunk_size, needles_batch)

        # 3. Шесть последовательных вызовов Ollama
        individual_responses = []
        total_ollama_duration = 0

        for idx, item in enumerate(needles_batch):
            question = item['q']
            context_for_this_q = retrieved_contexts_list[idx]

            mes = [
                {'role': 'system', 'content': system_answer}, 
                {"role": "user", "content": f"CONTEXT:\n{context_for_this_q}\n\nQUESTION: {question}"}
            ]
            
            payload = {
                "model": MODEL_NAME,
                "messages": mes,
                "stream": False,
                "options": {
                    "temperature": 0, 
                    "seed": 12345,
                    "num_ctx": token_size + 2000 # Запас под ответ
                }
            }
            
            start_t = time.time()
            response_data = request_with_retry(ollama_call, payload)
            total_ollama_duration += (time.time() - start_t)

            if response_data:
                ans = response_data['message']['content']
                ans_cleaned = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()
                individual_responses.append(f"{idx+1}. {question}: {ans_cleaned}")
            else:
                individual_responses.append(f"{idx+1}. {question}: [Ошибка ответа]")

        # Склеиваем ответы модели в один блок
        combined_model_response = "\n".join(individual_responses)

        # 4. ОДИН запрос к GPT-судье для оценки всего блока
        reference_data = ""
        for idx, item in enumerate(needles_batch):
            reference_data += f"{idx+1}. Question: {item['q']} | Correct Answer: {item['a']}\n"

        def gpt_judge_call(ref, model_out):
            return client.chat.completions.create(
                model=MODEL_JUDGE,
                messages=[
                    {"role": "system", "content": system_estimator},
                    {"role": "user", "content": f"REFERENCE DATA:\n{ref}\n\nMODEL OUTPUT:\n{model_out}"},
                ],
                temperature=0
            )

        start_time_gpt = time.time()
        gpt_response = request_with_retry(gpt_judge_call, reference_data, combined_model_response)
        duration_gpt = time.time() - start_time_gpt

        if gpt_response:
            raw_gpt_eval = gpt_response.choices[0].message.content.strip()
            # Ищем оценки [0] или [1]
            score_list = [int(x) for x in re.findall(r'\[([01])\]', raw_gpt_eval)]
            
            # Добиваем нулями, если GPT выдал меньше оценок, чем нужно
            while len(score_list) < len(needles_batch):
                score_list.append(0)
            score_list = score_list[:len(needles_batch)]
            
            found_count = sum(score_list)
        else:
            # Если GPT не ответил, помечаем задачу как невыполненную в existing
            with existing_lock: existing.discard(task_key)
            return []

        # 5. Формируем финальный результат
        results.append({
            "token_size": token_size,
            "chunk_size": chunk_size,
            "test_i": test_i,
            "found_count": found_count,
            "total_needles": len(needles_batch),
            "accuracy_percent": (found_count / len(needles_batch)) * 100,
            "time_ollama": total_ollama_duration,
            "time_evaluator": duration_gpt,
            "model_response": combined_model_response,      
            "evaluator_response": raw_gpt_eval,  
            "scores_raw": score_list             
        })
                    
    except Exception as e:
        print(f"Ошибка в тесте {test_i}: {e}")
        with existing_lock: existing.discard(task_key)
        traceback.print_exc()
        
    return results

def paralel_processing_multi(full_text_content: str):
    print(f"=== Запуск Multi-Needle RAG теста ===")
    global MAX_WORKERS
    with open('eval_tasks.json', 'r', encoding='utf-8') as f:
        eval_tasks = json.load(f)

    for c_size, t_size in zip(chunk_sizes, tokens):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            all_futures = []
            
            for task in eval_tasks:
                # Берем готовый пак из 6 иголок с уникальными вопросами
                needles_batch = task['needles'] 
                task_id = task['task_id']
                
                # Твоя логика с task_key и existing...
                task_key = (t_size, c_size, task_id)
                with existing_lock:
                    if task_key in existing: continue
                    existing.add(task_key)

                all_futures.append(
                    executor.submit(
                        process_multi_needle_task, 
                        full_text_content, 
                        needles_batch, 
                        c_size, 
                        t_size,
                        task_id
                    )
                )
            for future in as_completed(all_futures):
                batch_res = future.result() 
                if batch_res:
                    with results_lock:
                        key = (t_size, c_size)
                        if key not in results_dict:
                            results_dict[key] = []
                        results_dict[key].extend(batch_res)
                    
                    save_results()

def main():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:    
        data = json.load(f)
        full_text = data[0]["content"]
    paralel_processing_multi(full_text)

if __name__ == "__main__":
    main()