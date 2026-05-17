import json

MODEL = "gemma3"
with open(f'result/{MODEL}.json', 'r') as file:
    data = json.load(file)

total_time = 0
for d in data:
    for item in d:
        # Проверяем наличие ключей, так как последний блок в JSON — статистика без времени
        if "time_ollama" in item and "time_evaluator" in item:
            total_time += item["time_ollama"] + item["time_evaluator"]

print(f"Общее время: {total_time:.3f} сек.")

with open(f'rag_result/{MODEL}_rag_s.json', 'r') as file:
    data = json.load(file)

total_time = 0
for d in data:
    for item in d:
        # Проверяем наличие ключей, так как последний блок в JSON — статистика без времени
        if "time_ollama" in item and "time_evaluator" in item:
            total_time += item["time_ollama"] + item["time_evaluator"]

print(f"Общее время: {total_time:.3f} сек.")