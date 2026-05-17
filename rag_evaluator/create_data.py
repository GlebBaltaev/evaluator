import pandas as pd
import random
import json

def create_unique_tasks(csv_path, num_tasks=50, needles_per_task=6):
    # Читаем CSV
    df = pd.read_csv(csv_path)
    
    # Превращаем в список словарей для удобства
    all_needles = []
    for _, row in df.iterrows():
        all_needles.append({
            "inf": row['Иголка'],
            "q": row['Вопрос'],
            "a": row['Ответ']
        })

    final_tasks = []

    for i in range(num_tasks):
        selected_for_this_task = []
        used_questions = set()
        
        # Перемешиваем весь список иголок для каждого теста
        pool = all_needles.copy()
        random.shuffle(pool)
        
        for item in pool:
            if item['q'] not in used_questions:
                selected_for_this_task.append(item)
                used_questions.add(item['q'])
            
            # Как только набрали 6 уникальных вопросов — выходим
            if len(selected_for_this_task) == needles_per_task:
                break
        
        final_tasks.append({
            "task_id": i,
            "needles": selected_for_this_task
        })

    # Сохраняем в JSON
    with open('eval_tasks.json', 'w', encoding='utf-8') as f:
        json.dump(final_tasks, f, ensure_ascii=False, indent=4)
    
    print(f"Готово! Создано {len(final_tasks)} тестов. Проверь файл eval_tasks.json")

create_unique_tasks('../rag_needles.csv')