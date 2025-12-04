import json
import pandas as pd

tokens = [512, 1024, 2000, 4000, 8000]
percents = [0, 20, 40, 60, 80, 100]
NAME = "phi4_14b"
with open(f"result/{NAME}.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Создаем пустой DataFrame
df = pd.DataFrame(index=percents, columns=tokens)
print(df)
# Проходим по данным
for group in data:
    # берем последний элемент — это словарь с "percent"
    if isinstance(group, list):
        summary = group[-1]
        num_tokens = group[0]["num_tokens"]
        depth_percent = group[0]["depth_percent"]
        percent_value = summary["success_rate"]
        print(num_tokens, depth_percent)
        # Заполняем DataFrame
        df.loc[depth_percent, num_tokens] = percent_value

print(df)

# Сохраняем в CSV
df.to_csv(f"csv/{NAME}.csv", index=True, encoding="utf-8")