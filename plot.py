import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Увеличиваем общий размер шрифтов через rcParams для всех текстов
plt.rcParams.update({'font.size': 18}) 

# Список моделей и соответствующих файлов
models = ['dolphin3', 'llama', 'gemma3', 'phi4_14b']
tokens = [2000, 4000, 8000, 12000, 14000, 16000, 18000, 20000]

# Цвета для моделей
colors = {'dolphin3': '#1f77b4', 'gemma3': '#ff7f0e', 'llama': '#2ca02c', 'phi4_14b': '#d62728'}

# 1. Загрузка данных
data_frames = {}
for model in models:
    file_path = f"csv/{model}_rag.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        data_frames[model] = df
    else:
        print(f"Предупреждение: файл {file_path} не найден.")

# 2. Создаем сетку графиков n на m. Увеличил figsize для лучшей компоновки
fig, axes = plt.subplots(2, 4, figsize=(25, 12), sharey=True)
axes = axes.flatten()

for i, t in enumerate(tokens):
    ax = axes[i]
    col_str = str(t)
    
    for model in models:
        if model in data_frames and col_str in data_frames[model].columns:
            df_model = data_frames[model]
            # Увеличил linewidth до 3 и marker size до 8
            ax.plot(df_model.index, df_model[col_str], 
                    label=model, marker='o', markersize=8, 
                    color=colors[model], linewidth=3)
    
    # Заголовки графиков покрупнее
    ax.set_title(f"{t} tokens", fontsize=18, fontweight='bold', pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Настройка меток делений (ticks) — делаем их крупнее
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Подписи осей только для крайних графиков, чтобы не загромождать
    if i >= 4:
        ax.set_xlabel("Depth (%)", fontsize=18)
    if i % 4 == 0:
        ax.set_ylabel("Accuracy", fontsize=18)

# Добавляем ОБЩУЮ легенду сверху. Увеличил fontsize до 'xx-large'
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
           ncol=4, fontsize=30, frameon=True, shadow=True)

# Автоматическая корректировка расположения
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.savefig("rag.png", bbox_inches='tight', dpi=300)
plt.show()