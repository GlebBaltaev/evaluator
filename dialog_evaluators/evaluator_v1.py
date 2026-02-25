from openai import OpenAI
import json
import re
import time
import os
import requests

client = OpenAI(api_key="",
                base_url = "")

ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})

SERVER_URL = "http://81.94.150.224:11434/api/chat"
system_estimator = "You are a discrete estimator of how much the answer is correct based on the truly correct answer. You have an answered question and the right answer. Answer with the number 0 or 1, where 0 is the wrong answer and 1 is the correct answer and explain why do you think so in russian. Don't look at unnecessary words, look at the meaning, if everything is correct in the meaning, then this is 1. For example: if question is 'Какую книгу я счел лучшей?', the correct answear is 'биография'? so the answear 'Ты счел биографию самой лучшей книгой.' is correct answear and you must return 1 or question: 'Что я изучал и на каких инструментах играл?' correct answear: 'экономика, скрипка и фортепиано' so the answear: 'Ты учился в университете экономике и одновременно посещал музыкальную школу, где играл на скрипке и фортепиано. Какая у тебя самая любимая композиция для этих инструментов?' is also correct and on those similar cases you mast return 1. The answer may be more than the correct answer, unnecessary information, interpretation and additional information that is not in the correct answer, this is normal, as long as the words from the correct answer appear in the answer"


system_answear = "Answer the user's question. In answer don't ask something"
needles = [
    {
        "В прошлом году я жил в трех городах — Москве, Санкт-Петербурге и Казани, меняя работу каждые 4 месяца.": [
            "В каких городах я жил в прошлом году?",
            "Москва, Санкт-Петербург, Казань"
        ]
    },
    {
        "Моя любимая музыка — это джаз и классика, но на концерты я хожу только на джазовые.": [
            "Какие жанры музыки я люблю и на какие концерты хожу?",
            "люблю джаз и классику, хожу на концерты джаза"
        ]
    },
    {
        "Я читал три книги: детектив, научную фантастику и биографию. Биография мне понравилась больше всего.": [
            "Какую книгу я счел лучшей?",
            "биография"
        ]
    },
    {
        "У меня два питомца — кот по имени Том и собака по кличке Рыжик, которые не ладят между собой.": [
            "Какие животные у меня есть и как их зовут?",
            "кот Том, собака Рыжик"
        ]
    },
    {
        "На выходных я ездил то в горы, то на дачу, причем в горы — с друзьями, а на дачу — с семьей.": [
            "Куда я ездил на выходных и с кем?",
            "в горы с друзьями, на дачу с семьей"
        ]
    },
    {
        "У моего дедушки три внучки: Маша, Оля и Катя. Маша — старшая, Катя — младшая.": [
            "Сколько внучек у дедушки и какая из них старшая?",
            "три внучки, Маша — старшая"
        ]
    },
    {
        "Я учился в университете экономике и параллельно в музыкальной школе играл на скрипке и фортепиано.": [
            "Что я изучал и на каких инструментах играл?",
            "экономика, скрипка и фортепиано"
        ]
    },
    {
        "Погода в мае была переменчива — сначала шли дожди, потом на две недели выглянуло солнце.": [
            "Какая была погода в мае?",
            "сначала дожди, потом солнечно"
        ]
    },
    {
        "Мой друг Андрей — инженер, который работает над космическими проектами уже более 5 лет.": [
            "Чем занимается мой друг и сколько лет?",
            "инженер, космические проекты, более 5 лет"
        ]
    },
    {
        "Весной я посадил на балконе помидоры, базилик и укроп. Укроп рос лучше всего.": [
            "Какие растения я посадил на балконе и какое росло лучше?",
            "помидоры, базилик, укроп; лучше рос укроп"
        ]
    },
    {
        "В детстве я часто играл в футбол и баскетбол, но больше всего любил баскетбол.": [
            "Какие виды спорта я играл и какой был любимым?",
            "футбол, баскетбол; любимый — баскетбол"
        ]
    },
    {
        "Моя бабушка живет в деревне, где выращивает ягоды и овощи. Она часто приносит мне клубнику и помидоры.": [
            "Где живет бабушка и что она приносит?",
            "деревня; клубнику и помидоры"
        ]
    },
    {
        "Каждый день я ходил на работу пешком, кроме дождливых дней, тогда ехал на автобусе.": [
            "Как я добирался до работы?",
            "пешком, кроме дождливых дней — на автобусе"
        ]
    },
    {
        "Моя жена — врач, она работает в городской больнице.": [
            "Чем занимается моя жена и где работает?",
            "врач, городская больница"
        ]
    },
    {
        "В детстве я увлекался рисованием и лепкой из глины, но рисование было моим любимым занятием.": [
            "Чем я увлекался в детстве и что было любимым?",
            "рисование и лепка; любимое — рисование"
        ]
    },
    {
        "Мой брат живет в другой стране, в Канаде, и у него двое детей — сын и дочь.": [
            "Где живет мой брат и кто у него есть?",
            "Канада; сын и дочь"
        ]
    },
    {
        "Весной я езжу на велосипеде каждый день, устраиваю пикники и гуляю в парке.": [
            "Что я делаю весной?",
            "катаюсь на велосипеде, устраиваю пикники, гуляю в парке"
        ]
    },
    {
        "В музее я видел картины Айвазовского и Репина.": [
            "Какие картины я видел в музее?",
            "Айвазовского и Репина"
        ]
    },
    {
        "Я регулярно занимаюсь бегом и плаванием, но бегаю чаще, чем плаваю.": [
            "Какими видами спорта я занимаюсь и какой предпочитаю?",
            "бегом и плаванием; предпочитаю бег"
        ]
    },
    {
        "Моя семья любит путешествовать в Европу, особенно во Францию и Италию.": [
            "Куда любит путешествовать моя семья?",
            "Европа, особенно Франция и Италия"
        ]
    }
]
system_estimator = "You are a discrete estimator of how much the answer is correct based on the truly correct answer. You have an answered question and the right answer. Answer with the number 0 or 1, where 0 is the wrong answer and 1 is the correct answer and explain why do you think so in russian. Don't look at unnecessary words, look at the meaning, if everything is correct in the meaning, then this is 1. For example: if question is 'Какую книгу я счел лучшей?', the correct answear is 'биография'? so the answear 'Ты счел биографию самой лучшей книгой.' is correct answear and you must return 1 or question: 'Что я изучал и на каких инструментах играл?' correct answear: 'экономика, скрипка и фортепиано' so the answear: 'Ты учился в университете экономике и одновременно посещал музыкальную школу, где играл на скрипке и фортепиано. Какая у тебя самая любимая композиция для этих инструментов?' is also correct and on those similar cases you mast return 1. The answer may be more than the correct answer, unnecessary information, interpretation and additional information that is not in the correct answer, this is normal, as long as the words from the correct answer appear in the answer"


system_answear = "Answer the user's question. In answer don't ask something"


tokens = [512, 1024, 2000, 4000, 8000]

percents = [0, 20, 40, 60, 80, 100]
tests = []
results = []


MAX_RETRIES = 5
RETRY_DELAY = 5  # секунд
MODEL_NAME = "gemma3_12b"
OUTPUT_FILE = f'result/{MODEL_NAME}1.json'
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
    (r["num_tokens"], r["depth_percent"], r["needle"])
    for block in results for r in block if "needle" in r
}

def save_results():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def request_with_retry(func, *args, **kwargs):
    """Выполняет функцию с ретраями"""
    time.sleep(RETRY_DELAY)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Ошибка (попытка {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None

for token in tokens:
    for percent in percents:
        result = []
        for elem in needles:
            for inf, q_a in elem.items():
                key = (token, percent, inf)
                if key in existing:
                    print(f"Пропускаю {key} (уже есть в JSON)")
                    continue

                # Загружаем базовый контент
                with open(f"text_needles/text_{token}_{percent}.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    if item['content'] == '0xFFFF':
                        item['content'] = inf
                data = data + [{"role": "user", "content": q_a[0]}]
                mes = [{'role': 'system', 'content': system_answear}] + data
                payload = {
                    "model": MODEL_NAME,
                    "messages": mes,
                    "temperature": 0,
                    "stream": False
                }

                # --- Запрос в ollama ---
                def ollama_call():
                    resp = ollama_session.post(SERVER_URL, data=json.dumps(payload), timeout=120)
                    resp.raise_for_status()
                    return resp.json()

                response_data = request_with_retry(ollama_call)
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
                        model="gpt-4.1",
                        messages=[
                            {"role": "system", "content": system_estimator},
                            #{"role": "user", "content": f"Question:{q_a[1]} \n Answer: {cleaned_text} \n Correct answer: {q_a[1]} "},
                            {"role": "user", "content": f"Question:{q_a[0]} Truly correct answer: {q_a[1]} \n Answer that you need to estimate: {cleaned_text} \n "},

                        ]
                    )

                gpt_response = request_with_retry(gpt_call)
                if gpt_response is None:
                    print(f"Пропускаю {key}: GPT не ответил")
                    continue

                found = int(gpt_response.choices[0].message.content[0])
                result.append({
                    "num_tokens": token,
                    "depth_percent": percent,
                    "needle": inf,
                    "response": cleaned_text,
                    "found": found
                })

                print(gpt_response.choices[0].message.content)

                existing.add(key)

        if result:  # сохраняем блок только после всех needle
            result.append({
                "overall": sum(d["found"] for d in result),
                "percent": sum(d["found"] for d in result) / len(result)
            })
            results.append(result)
            save_results()
