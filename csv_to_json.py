import json
import csv


NEEDLES_FILE_PATH = "text_needles.csv"
RESULT_JSON_PATH = "text_needles/text_needles.json"

final_json = []
with open(NEEDLES_FILE_PATH, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        final_json.append({row[0]: [row[1], row[2]]})
with open(RESULT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(final_json, f, ensure_ascii=False, indent=4)
    