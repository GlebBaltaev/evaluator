import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Устанавливаем неинтерактивный режим для matplotlib
plt.switch_backend('Agg')

# Функция для вычисления среднего времени на запрос
def calculate_avg_time(filename, parallel_count):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Если последний элемент - число (время), удаляем его
        if isinstance(data[-1], (int, float)):
            data = data[:-1]
        
        total_time = 0
        total_requests = 0
        
        # Обрабатываем структуру вложенных списков
        for outer_list in data:
            if isinstance(outer_list, list):
                for item in outer_list:
                    if isinstance(item, dict) and 'time_ollama' in item:
                        total_time += item['time_ollama']
                        total_requests += 1
        
        # Обычное среднее время на запрос
        avg_time_per_request = total_time / total_requests if total_requests > 0 else 0
        
        # Среднее время с учетом параллельных запросов (делим на количество параллельных запросов)
        avg_time_with_parallel = total_time / (total_requests * parallel_count) if total_requests > 0 else 0
        
        return avg_time_per_request, avg_time_with_parallel, total_requests
    except Exception as e:
        print(f"Ошибка при обработке {filename}: {e}")
        return None, None, 0

# Список файлов и соответствующих параллельных запросов
# files = [
#     ('result/phi4_14b_T4_default.json', 1),
#     ('result/phi4_14b_PARALEL_2_work.json', 2),
#     ('result/phi4_14b_T4_PARALEL_4_work.json', 4),
#     ('result/phi4_14b_PARALEL_10_work.json', 10),
#     ('result/phi4_14b_PARALEL_20_work.json', 20)
# ]

files = [
    ('result/phi4_14b_20_work.json', 10),
    ('result/phi4_14b_batch_20_work.json', 10),
    ('result/phi4_14b_par_20_work.json', 10),
]

label = ['old_par', 'batch', 'new_par'] 
# Собираем данные
parallel_requests = []
avg_times_per_request = []  # Обычное среднее (только на количество запросов)
avg_times_with_parallel = []  # Среднее с учетом параллельных запросов
total_requests_list = []

for filename, parallel_count in files:
    avg_per_request, avg_with_parallel, total_requests = calculate_avg_time(filename, parallel_count)
    if avg_per_request is not None:
        parallel_requests.append(parallel_count)
        avg_times_per_request.append(avg_per_request)
        avg_times_with_parallel.append(avg_with_parallel)
        total_requests_list.append(total_requests)
        print(f"{filename}: {parallel_count} параллельных запросов, {total_requests} всего запросов")
        print(f"  Обычное среднее: {avg_per_request:.2f} сек на запрос")
        print(f"  С учетом параллельности: {avg_with_parallel:.2f} сек на запрос")

# Создаем теплокарты для обоих типов средних значений

# 1. Теплокарта для обычного среднего
plt.figure(figsize=(12, 3))
heatmap_data_normal = np.array([avg_times_per_request])

sns.heatmap(heatmap_data_normal, 
            annot=True, 
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Время (секунды)'},
            xticklabels=label,
            yticklabels=['Время'])

plt.xlabel('Количество параллельных запросов')
plt.ylabel('Среднее время на запрос')
plt.title('Теплокарта: Обычное среднее время выполнения (деление на количество запросов)')
plt.tight_layout()
plt.savefig('heatmap_normal_avg.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Теплокарта для среднего с учетом параллельных запросов
plt.figure(figsize=(12, 3))
heatmap_data_parallel = np.array([avg_times_with_parallel])

sns.heatmap(heatmap_data_parallel, 
            annot=True, 
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Время (секунды)'},
            xticklabels=label,
            yticklabels=['Время'])

plt.xlabel('Количество параллельных запросов')
plt.ylabel('Среднее время на запрос')
plt.title('Теплокарта: Среднее время с учетом параллельности (деление на запросы × параллельность)')
plt.tight_layout()
plt.savefig('heatmap_parallel_avg.png', dpi=300, bbox_inches='tight')
plt.close()



for i in range(len(parallel_requests)):
    print(f"{parallel_requests[i]:<15} {total_requests_list[i]:<10} {avg_times_per_request[i]:<20.2f} {avg_times_with_parallel[i]:<25.2f}")

print("\nГрафики сохранены в файлы:")
print("- plot_heatmap.png (обычное среднее)")
print("- plot_heatmap_avg.png (с учетом параллельности)")