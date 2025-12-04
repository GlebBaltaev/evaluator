from openai import OpenAI
import json
import re
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple, Dict, Any

ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})

SERVER_URL = "http://localhost:11434/api/chat"

client = OpenAI(api_key="",
                base_url = "")

with open('needles.json', 'r') as file:
    needles = json.load(file)
    
system_estimator = """You are a discrete estimator of how much the answer is correct based on the truly correct answer.
You have an answered question and the right answer. Answer with the number 0 or 1, where 0 is the wrong answer and 1 is the correct answer and explain 
why do you think so in russian. Don't look at unnecessary words, look at the meaning, if everything is correct in the meaning, then this is 1. 
For example: if question is 'Какую книгу я счел лучшей?', the correct answear is 'биография'? so the answear 'Ты счел биографию самой лучшей книгой.' 
is correct answear and you must return 1 or question: 'Что я изучал и на каких инструментах играл?' correct answear: 'экономика, скрипка и фортепиано' 
so the answear: 'Ты учился в университете экономике и одновременно посещал музыкальную школу, где играл на скрипке и фортепиано. 
Какая у тебя самая любимая композиция для этих инструментов?' is also correct and on those similar cases you mast return 1.
Another example: 'Password: “WarCraft”' and if the answer is 'Нет, ты мне не называл свой пароль от игры. Я не знаю, какой у тебя пароль.',
it's wrong because the password was given to “WarCraft” (even put in quotation marks, so it's definitely the password) so that is 0. 
The answer may be more than the correct answer, unnecessary information, interpretation and additional information that is not in the correct answer, 
this is normal, as long as the words from the correct answer appear in the answer"""


system_answer = "Answer the user's question. In answer don't ask something. If you don`t know answer, say that you don`t know answer because text hasn`t got information about this. Question: "


tokens = [512]
percents = [0, 20, 40, 60, 100]

MAX_RETRIES = 5
RETRY_DELAY = 5  # секунд
MODEL_NAME = "phi4_14b"
OUTPUT_FILE = f'result/{MODEL_NAME}_par_20_work.json'
MAX_WORKERS = 10  # Максимальное количество параллельных запросов

# Потокобезопасные структуры данных
results_lock = threading.Lock()
existing_lock = threading.Lock()

# Загружаем уже имеющиеся результаты
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = []
else:
    results = []

# Множество для проверки дубликатов (потокобезопасное)
with existing_lock:
    existing = {
        (r["num_tokens"], r["depth_percent"], r["needle"], r.get("question"))
        for block in results for r in block if "needle" in r
    }

def save_results():
    """Потокобезопасное сохранение результатов"""
    with results_lock:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

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
    resp = ollama_session.post(SERVER_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()
def process_single_task(token: int, percent: float, elem: Dict) -> List[Dict]:
    """Обрабатывает одну задачу: token, percent, needle"""
    result = []
    
    for inf, q_a in elem.items():
        key = (token, percent, inf, q_a[0])
        
        # Проверяем, не обрабатывалась ли уже эта задача
        with existing_lock:
            if key in existing:
                print(f"Пропускаю {key} (уже есть в JSON)")
                continue
            # Временно помечаем как обрабатываемую
            existing.add(key)
    
        try:
            # Загружаем базовый контент
            with open(f"text_needles/text_{token}_{percent}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if item['content'] == '0xFFFF':
                    item['content'] = inf
            
            mes = data + [{'role': 'system', 'content': system_answer + q_a[0]}]
            payload = {
                "model": MODEL_NAME,
                "messages": mes,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed": 12345,
                }
            }
            
            # Запрос в ollama
            start_ollama = time.time()
            response_data = request_with_retry(ollama_call, payload)
            end_ollama = time.time()
            
            if response_data is None:
                print(f"Пропускаю {key}: ollama не ответил")
                # Удаляем из обрабатываемых, если не удалось
                with existing_lock:
                    existing.remove(key)
                continue

            if 'message' in response_data and 'content' in response_data['message']:
                pattern = r"<think>.*?</think>"
                cleaned_text = re.sub(pattern, "", response_data['message']['content'], flags=re.DOTALL)
            else:
                print("Ошибка: нет ключа 'message' или 'content'.")
                print("Полный ответ:", response_data)
                with existing_lock:
                    existing.remove(key)
                continue

            print(f"Обработано: token={token}, percent={percent}, needle={inf[:50]}...")
            
            task_result = {
                "num_tokens": token,
                "depth_percent": percent,
                "needle": inf,
                "question": q_a[0],
                "response": cleaned_text,
                "time_ollama": end_ollama - start_ollama,
            }
            
            result.append(task_result)
            
        except Exception as e:
            print(f"Ошибка при обработке {key}: {e}")
            # Удаляем из обрабатываемых при ошибке
            with existing_lock:
                if key in existing:
                    existing.remove(key)
    
    return result
def paralel_processing():
    """Непрерывная очередь"""
    print("=== Непрерывная очередь ===")
    time_start = time.time()
    
    # Создаем список всех задач
    tasks = []
    for token in tokens:
        for percent in percents:
            for elem in needles:
                tasks.append((token, percent, elem))
    
    # Обрабатываем задачи параллельно с максимальным количеством воркеров = 10
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Отправляем все задачи в executor
        future_to_task = {
            executor.submit(process_single_task, token, percent, elem): (token, percent, elem)
            for token, percent, elem in tasks
        }
        
        # Собираем результаты по мере завершения
        for future in as_completed(future_to_task):
            token, percent, elem = future_to_task[future]
            try:
                task_results = future.result()
                if task_results:
                    # Добавляем результаты в общий список
                    with results_lock:
                        results.append(task_results)
                    # Сохраняем после каждой завершенной группы
                    save_results()
                    print(f"Сохранены результаты для token={token}, percent={percent}")
                    
            except Exception as exc:
                print(f'Задача сгенерировала исключение: {exc}')
    
    total_time = time.time() - time_start
    print(f"Общее время выполнения (Подход 1): {total_time:.2f} секунд")
    
    # Добавляем общее время в результаты
    results.append(f"Total time Approach 1: {total_time}")
    save_results()

def batch_processing():
    """ Батчевая обработка """
    print("=== Батчевая обработка ===")
    time_start = time.time()
    
    # Создаем список всех задач
    tasks = []
    for token in tokens:
        for percent in percents:
            for elem in needles:
                tasks.append((token, percent, elem))
    
    # Разбиваем задачи на батчи по 10
    batch_size = 10
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    print(f"Всего задач: {len(tasks)}, разбито на {len(batches)} батчей по {batch_size}")
    
    for batch_num, batch_tasks in enumerate(batches, 1):
        print(f"\n--- Обрабатываю батч {batch_num}/{len(batches)} ---")
        
        # Создаем временный пул для текущего батча
        with ThreadPoolExecutor(max_workers=len(batch_tasks)) as executor:
            # Отправляем все задачи текущего батча
            future_to_task = {
                executor.submit(process_single_task, token, percent, elem): (token, percent, elem)
                for token, percent, elem in batch_tasks
            }
            
            # Ждем завершения ВСЕХ задач в текущем батче
            completed_in_batch = 0
            for future in as_completed(future_to_task):
                token, percent, elem = future_to_task[future]
                completed_in_batch += 1
                try:
                    task_results = future.result()
                    if task_results:
                        with results_lock:
                            results.append(task_results)
                        print(f"Завершено {completed_in_batch}/{len(batch_tasks)} в батче {batch_num}")
                        
                except Exception as exc:
                    print(f'Задача сгенерировала исключение: {exc}')
            
            # Сохраняем результаты после каждого батча
            save_results()
            print(f"Батч {batch_num} завершен. Результаты сохранены.")
    
    total_time = time.time() - time_start
    print(f"Общее время выполнения (Подход 2): {total_time:.2f} секунд")
    
    # Добавляем общее время в результаты
    results.append(f"Total time Approach 2: {total_time}")
    save_results()

def main():
    
    # ПОДХОД 2: Батчевая обработка  
    paralel_processing()

if __name__ == "__main__":
    main()