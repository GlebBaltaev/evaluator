from openai import OpenAI
import json
import re
import time
import os
import requests
import subprocess

env = os.environ.copy()
env["OLLAMA_NUM_PARALLEL"] = "1"
subprocess.run(["ollama", "serve"], env=env)

client = OpenAI(api_key="",
                base_url = "")

ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})

SERVER_URL = "http://localhost:11434/api/chat"


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


tokens = [512, 1024, 2000, 4000, 8000, 12000, 14000, 16000, 18000, 20000]

percents = [0, 20, 40, 60, 80, 100]
tests = []
results = []


MAX_RETRIES = 5
RETRY_DELAY = 5  # секунд
MODEL_NAME = "phi3_14b"
OUTPUT_FILE = f'result/{MODEL_NAME}_system_in_end.json'
# Загружаем уже имеющиеся результаты
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = []
else:
    results = []

# Индекс для проверки дубликатов
existing = {
    (r["num_tokens"], r["depth_percent"], r["needle"], r.get("question"))
    for block in results for r in block if "needle" in r
}

def save_results():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def request_with_retry(func, *args, **kwargs):
    """Выполняет функцию с ретраями"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Ошибка (попытка {attempt}): {e}")
    return None

for token in tokens:
    for percent in percents:
        result = []
        for elem in needles:
            for inf, q_a in elem.items():
                key = (token, percent, inf, q_a[0])  # добавляем вопрос
                if key in existing:
                    print(f"Пропускаю {key} (уже есть в JSON)")
                    continue
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
                        "seed": 123
                    }
                    
                }
                # --- Запрос в ollama ---
                def ollama_call():
                    resp = ollama_session.post(SERVER_URL, json=payload, timeout=120)
                    resp.raise_for_status()
                    return resp.json()
                start_ollama = time.time()
                response_data = request_with_retry(ollama_call)
                end_ollama = time.time()
                if response_data is None:
                    print(f"Пропускаю {key}: ollama не ответил")
                    continue

                if 'message' in response_data and 'content' in response_data['message']:
                    pattern = r"<think>.*?</think>"
                    cleaned_text = re.sub(pattern, "", response_data['message']['content'], flags=re.DOTALL)
                else:
                    print("Ошибка: нет ключа 'message' или 'content'.")
                    print("Полный ответ:", response_data)
                    continue  # не идём в GPT

                print(token, percent, cleaned_text)

                # --- Запрос в GPT ---
                def gpt_call():
                    return client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[
                            {"role": "system", "content": system_estimator},
                            {"role": "user", "content": f"Question:{q_a[0]} Truly correct answer: {q_a[1]} \n Answer that you need to estimate: {cleaned_text} \n "},
                        ]
                    )
                start_eval = time.time()
                gpt_response = request_with_retry(gpt_call)
                end_eval = time.time()
                if gpt_response is None:
                    print(f"Пропускаю {key}: GPT не ответил")
                    continue

                content = gpt_response.choices[0].message.content.strip()

                match = re.search(r'\b[01]\b', content)
                if match:
                    found = int(match.group(0))
                result.append({
                    "num_tokens": token,
                    "depth_percent": percent,
                    "needle": inf,
                    "question": q_a[0],
                    "response": cleaned_text,
                    "found": found,
                    "time_ollama": end_ollama - start_ollama,
                    "time_evaluator": end_eval - start_eval
                })

                print(gpt_response.choices[0].message.content)

                existing.add(key)

        if result: 
            result.append({
                "overall": sum(d["found"] for d in result),
                "percent": sum(d["found"] for d in result) / len(result)
            })
            results.append(result)
            save_results()
