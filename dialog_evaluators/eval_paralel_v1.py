from openai import OpenAI
import json
import re
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Глобальные переменные и конфигурация
ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})

SERVER_URL = "http://localhost:11434/api/chat"

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
    },
    {
        "Я родился 1 сентября 1975 года": [
            "Когда я родился?",
            "1 сент.1975 года"
        ]
    },
    {
        "Вчера был день святого Валентина. Я подарил жене попугая": [
            "Что я подарил жене не день Святого Валентина",
            "Попугай"
        ]
    },
    {
        "На новый год я взял отпуск, который провел на Кубе": [
            "Где я провел отпуск?",
            "На Кубе"
        ]
    },
    {
        "В детстве я ходил в музыкальную школу и играл на баяне.": [
            "На каком музыкальном инструменте я играл в детстве?",
            "Баян"
        ]
    },
    {
        "В детстве я много читал. Особенно мне Рассказ Чехова - Каштанка. Я ее читал 4 раза": [
            "Сколько раз я читал Каштанку",
            "4 раза"
        ]
    },
    {
        "Каштанку я читал 4 раза": [
            "Сколько раз читал Каштанку?",
            "4 раза"
        ]
    },
    {
        "password: “король”": [
            "Я тебе писал пароль. Назови его?",
            "король"
        ]
    },
    {
        "Пароль: “WarCraft”": [
            "Я забыл свой пароль от игры. Я тебе называл его?",
            "WarCraft"
        ]
    },
    {
        "Я терпеть не могу гороховый суп. Я его не ем": [
            "Какой суп я не ем?",
            "Гороховый"
        ]
    },
    {
        "Я терпеть не могу гороховый суп. Я его не ем": [
            "Назови блюда, которые я не ем?",
            "Гороховый суп"
        ]
    },
    {
        "Моему сыну 10 лет. Его зовут Василий.": [
            "Что ты знаешь о моем сыне?",
            "Сын Василий, ему 10 лет"
        ]
    },
    {
        "Моему сыну 10 лет. Его зовут Василий.": [
            "Сколько лет моему сыну?",
            "10 лет"
        ]
    },
    {
        "Моему сыну 10 лет. Его зовут Василий.": [
            "Как зовут моего сына?",
            "Василий"
        ]
    },
    {
        "Контрольное слово: ТАРАКАН.": [
            "Назови контрольное слово.",
            "ТАРАКАН"
        ]
    },
    {
        "Контрольное слово: ТАРАКАН.": [
            "Назови контрольное слово.",
            "Таракан"
        ]
    },
    {
        "В детсве я любил спорт и играл в хоккей, в юности студентом я играл на рояле, сейчас я больше увлекаюсь игрой в покер.": [
            "Какие хобби у меня были",
            "Хоккей, рояль, покер"
        ]
    },
    {
        "Делу время, потехе час": [
            "Помнишь я тебе говорил однажды пословицу? напомни мне пожалуйста.",
            "Делу время, потехе час"
        ]
    },
    {
        "Моего друга зовут Александр Пушкин. Но он не писатель. Все, что он пишет, это подписи на кредитных договорах.": [
            "Что пишет мой друг Александр Пушкин.",
            "Подписи на кредитных договорах"
        ]
    },
    {
        "глокая куздра штеко будланула бокра и курдячит бокрёнка": [
            "Кого будланула глокая куздра?",
            "бокра"
        ]
    }
]
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
percents = [0, 20, 40, 60, 80, 100]

MAX_RETRIES = 5
RETRY_DELAY = 5  # секунд
MODEL_NAME = "phi4_14b"
OUTPUT_FILE = f'result/{MODEL_NAME}_20_work.json'
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

def process_single_task(token, percent, elem):
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

def main():
    """Основная функция с параллельной обработкой"""
    time_start = time.time()
    
    # Создаем список всех задач
    tasks = []
    for token in tokens:
        for percent in percents:
            for elem in needles:
                tasks.append((token, percent, elem))
    
    # Обрабатываем задачи параллельно
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    print(f"Общее время выполнения: {total_time:.2f} секунд")
    
    # Добавляем общее время в результаты
    results.append(f"Total time: {total_time}")
    save_results()

if __name__ == "__main__":
    main()