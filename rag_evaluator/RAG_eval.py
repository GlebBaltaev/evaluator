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
from create_vdb import run_rag_needle_test

ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})



client = OpenAI(api_key="",
                base_url = "")



with open('../text_needles/text_needles.json', 'r') as file:
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
this is normal, as long as the words from the correct answer appear in the answer. Write your rating (0 or 1) at the end of the sentence after your explanation."""


system_answer = "Answer the user's question. In answer don't ask something. If you don`t know answer, say that you don`t know answer because text hasn`t got information about this. Do not pay attention to the fact that the answer is not real in life, if it is written in the text, then it is so."


tokens = [2000, 4000, 8000, 12000, 14000, 16000, 18000, 20000]
percents = [0, 20, 40, 60, 80, 100]

MAX_RETRIES = 1
RETRY_DELAY = 5  # секунд
MODEL_NAME = "phi4_14b"
MODEL_JUDGE = "deepseek-chat"
OUTPUT_FILE = f'../rag_result/{MODEL_NAME}_rag.json'
MAX_WORKERS = 1  # Максимальное количество параллельных запросов
SERVER_URL = "http://localhost:11434/api/chat"
JUDGE_URL = ""


# Потокобезопасные структуры данных
results_lock = threading.Lock()
existing_lock = threading.Lock()

results_dict = {}

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
            for block in raw_data:
                if isinstance(block, list) and len(block) > 0:
                    # Ищем первый элемент, который не является статистикой, чтобы взять ключи
                    first_task = next((item for item in block if item.get("block_type") != "statistics"), None)
                    if first_task:
                        key = (first_task["num_tokens"], first_task["depth_percent"])
                        # Очищаем блок от старой статистики перед загрузкой, 
                        # мы пересчитаем её заново при сохранении
                        results_dict[key] = [item for item in block if item.get("block_type") != "statistics"]
        except json.JSONDecodeError:
            results_dict = {}
with existing_lock:
    existing = set()
    for key, items in results_dict.items():
        for r in items:
            # Добавляем в кэш пропусков
            existing.add((r["num_tokens"], r["depth_percent"], r["needle"], r["question"]))
print(f"Загружено из файла: {len(existing)} готовых результатов.")

def save_results():
    """Потокобезопасное сохранение с сортировкой блоков и статистикой по ОДНОЙ иголке"""
    final_to_save = []
    
    with results_lock:
        # Сортируем ключи: сначала токены, потом размер чанка
        sorted_keys = sorted(results_dict.keys(), key=lambda x: (x[0], x[1]))
        
        for (t, p) in sorted_keys:
            items = results_dict[(t, p)]
            
            # total_needles — это сколько всего иголок мы планировали протестировать в этой точке
            total_expected = len(needles) 
            processed = len(items)
            found = sum(i.get('found', 0) for i in items)
            
            stats = {
                "block_type": "statistics", 
                "token_size": t, 
                "depth_percent": p, 
                "completed_needles": processed, 
                "total_needles_in_csv": total_expected,
                "total_found": found, 
                "success_rate_percent": found / total_expected
            }
            
            # Сортируем иголки внутри блока по алфавиту или по порядку (если есть id)
            sorted_items = sorted(items, key=lambda x: x.get('needle', ""))
            
            block_with_stats = sorted_items + [stats]
            final_to_save.append(block_with_stats)
            
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
    resp = ollama_session.post(SERVER_URL, json=payload, timeout=1000)
    resp.raise_for_status()
    return resp.json()

def ollama_judge_call(payload):
    """Функция для вызова Ollama в качестве оценщика"""
    resp = ollama_session.post(JUDGE_URL, json=payload, timeout=1000)
    resp.raise_for_status()
    return resp.json()

def process_single_task(token: int, percent: float, elem: Dict) -> List[Dict]:
    """Обрабатывает одну задачу и сразу проводит оценку через оценщик"""
    result = []
    
    for inf, q_a in elem.items():
        key = (token, percent, inf, q_a[0])
        
        with existing_lock:
            if key in existing:
                print(f"Пропускаю {key} (уже есть в JSON)")
                continue
            existing.add(key)
    
        try:
            # 1. Загрузка контента
            with open(f"../data_rag/data_rag_{percent}.json", 'r', encoding='utf-8') as f:
                data_json = json.load(f)
                # Берем контент (поддержка формата [ {"content": ...} ])
                content_text = data_json[0]['content'] if isinstance(data_json, list) else data_json['content']
            
            # Подстановка иголки
            content_text = content_text.replace('0xFFFF', '\n' + inf + '\n')
            retrieved_contexts_list = run_rag_needle_test(content_text, token, {'q': q_a[0]})
            # Формируем промпт для Ollama
            mes = [{'role': 'system', 'content': system_answer}, 
                   {"role": "user", "content": f"CONTEXT:\n{retrieved_contexts_list}\n\nQUESTION: {q_a[0]}"}]
            
            payload = {
                "model": MODEL_NAME,
                "messages": mes,
                "stream": False,
                "options": {"temperature": 0, "seed": 12345}
            }
            
            # 2. Запрос в Ollama (Исполнитель)
            start_ollama = time.time()
            response_data = request_with_retry(ollama_call, payload)
            end_ollama = time.time()
            
            if response_data is None:
                with existing_lock: existing.remove(key)
                continue

            # Очистка от <think>
            raw_response = response_data['message']['content']
            cleaned_text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

            # 3. Запрос в Оценщик
            print(f"Оцениваю ответ для: {inf[:30]}...")
            
            #Оценщик - opensource модель
            # payload = {
            #     "model": MODEL_JUDGE,
            #     "messages": [
            #             {"role": "system", "content": system_estimator},
            #             {"role": "user", "content": f"Question: {q_a[0]} Truly correct answer: {q_a[1]} \n Answer that you need to estimate: {cleaned_text}"},
            #         ],
            #     "stream": False,
            #     "options": {"temperature": 0, "seed": 12345}
            # }
            # start_eval = time.time()
            # response_data = request_with_retry(ollama_judge_call, payload)
            # end_eval = time.time()
            
            # if response_data:
            #     raw_response = response_data['message']['content']
            #     eval_content = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
            #     eval_content = gpt_response.choices[0].message.content.strip()
                
            #     # Ищем цифру 0 или 1 в самом конце строки, допуская возможные точки или пробелы после
            #     match = re.search(r'([01])\s*\.?$', eval_content)
                
            #     if match:
            #         found = int(match.group(1))
            #     else:
            #         # Если в конце не найдено, пробуем найти хоть где-то, как запасной вариант
            #         fallback_match = re.search(r'\b[01]\b', eval_content)
            #         found = int(fallback_match.group(0)) if fallback_match else 0
            # else:
            #     found = 0
            #     eval_content = "Ошибка вызова оценщика"

            # Оценщик - chatGPT или deepseek
            def gpt_call():
                return client.chat.completions.create(
                    model=MODEL_JUDGE,
                    messages=[
                        {"role": "system", "content": system_estimator},
                        {"role": "user", "content": f"Question: {q_a[0]} Truly correct answer: {q_a[1]} \n Answer that you need to estimate: {cleaned_text}"},
                    ]
                )
            
            start_eval = time.time()
            gpt_response = request_with_retry(gpt_call)
            end_eval = time.time()
            
            if gpt_response:
                eval_content = gpt_response.choices[0].message.content.strip()
                
                # Ищем цифру 0 или 1 в самом конце строки, допуская возможные точки или пробелы после
                match = re.search(r'([01])\s*\.?$', eval_content)
                
                if match:
                    found = int(match.group(1))
                else:
                    # Если в конце не найдено, пробуем найти хоть где-то, как запасной вариант
                    fallback_match = re.search(r'\b[01]\b', eval_content)
                    found = int(fallback_match.group(0)) if fallback_match else 0
            else:
                found = 0
                eval_content = "Ошибка вызова GPT"

            # Формируем результат иголки
            task_result = {
                "num_tokens": token,
                "depth_percent": percent,
                "needle": inf,
                "question": q_a[0],
                "expected": q_a[1],
                "response": cleaned_text,
                "found": found,
                "explanation": eval_content,
                "time_ollama": end_ollama - start_ollama,
                "time_evaluator": end_eval - start_eval
            }
            
            result.append(task_result)
            print(f"Готово: {token}_{percent} | Оценка: {found}")
            
        except Exception as e:
            print(f"Ошибка при обработке {key}: {e}")
            traceback.print_exc()
            with existing_lock:
                if key in existing: existing.remove(key)
    
    return result

def paralel_processing():
    """Непрерывная очередь с группировкой по точкам"""
    print("=== Paralel processing ===")
    time_start = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor: 
        for token in tokens:
            for percent in percents:
                print(f"\n Начинаю обработку точки: {token} tokens, {percent}% depth")
                
                # Создаем список задач только для ТЕКУЩЕЙ точки
                futures = [executor.submit(process_single_task, token, percent, elem) for elem in needles]
                
                new_results_for_point = []
                
                for future in as_completed(futures):
                    try:
                        task_results = future.result()
                        if task_results:
                            new_results_for_point.extend(task_results)
                    except Exception as exc:
                        print(f'Ошибка в задаче: {exc}')

                # Когда иголки для (token, percent) обработаны — мерджим их в общий словарь
                if new_results_for_point:
                    with results_lock:
                        key = (token, percent)
                        if key not in results_dict:
                            results_dict[key] = []
                        
                        # Хэш-сет для проверки уникальности внутри текущего блока в памяти
                        current_keys = { (r["needle"], r["question"]) for r in results_dict[key] }
                        
                        for r in new_results_for_point:
                            # Добавляем только если такой иголки еще нет в этом блоке
                            if (r["needle"], r["question"]) not in current_keys:
                                results_dict[key].append(r)

                    # Сохраняем всё состояние (пересчитает статистику для всех блоков)
                    save_results()
                    print(f"--- Точка {token}_{percent} обновлена и сохранена. ---")

    total_time = time.time() - time_start
    print(f"Общее время выполнения: {total_time:.2f} секунд")


def main():
    paralel_processing()

if __name__ == "__main__":
    main()