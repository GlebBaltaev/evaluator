import pdfplumber
import pytesseract
from PIL import Image
import os

def extract_text_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

text = extract_text_pdfplumber("sample.pdf")
print(text)
with open("sample.txt", 'w') as file:
    file.write(text)

# def extract_text_blocks(page):
#     """
#     Возвращает список bounding-box'ов текстовых блоков,
#     сгруппированных построчно.
#     """
#     words = page.extract_words()
#     if not words:
#         return []

#     # Группировка слов в строки по Y-координате
#     lines = {}
#     for w in words:
#         # нормализация, чтобы слова одной строки были в одном ключе
#         key = round(w["top"] / 5)
#         lines.setdefault(key, []).append(w)

#     # Создаем bounding-box для каждой строки
#     blocks = []
#     for _, words_line in sorted(lines.items()):
#         x0 = min(w["x0"] for w in words_line)
#         top = min(w["top"] for w in words_line)
#         x1 = max(w["x1"] for w in words_line)
#         bottom = max(w["bottom"] for w in words_line)
#         blocks.append((x0, top, x1, bottom))

#     return blocks


# def ocr_pdf_text_only(pdf_path, output_path="output.txt", dpi=300):
#     result_text = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, page in enumerate(pdf.pages, start=1):
#             print(f"[+] Обработка страницы {page_num}/{len(pdf.pages)}")

#             blocks = extract_text_blocks(page)
#             if not blocks:
#                 continue

#             page_lines = []

#             for bbox in blocks:
#                 try:
#                     # Вырезаем текстовый фрагмент
#                     cropped = page.within_bbox(bbox)
#                     img = cropped.to_image(resolution=dpi).original.convert("RGB")

#                     # OCR только по выделенной области
#                     line_text = pytesseract.image_to_string(img, lang="rus", config="--psm 6")
#                     line_text = line_text.strip()

#                     if line_text:
#                         page_lines.append(line_text)
#                 except Exception as e:
#                     print(f"Ошибка OCR блока: {e}")
#                     continue

#             result_text.append("\n".join(page_lines))

#     # Сохраняем результат
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write("\n\n".join(result_text))

#     print(f"\nГотово! Текст сохранён в: {output_path}")


# # ---------- ЗАПУСК ----------
# if __name__ == "__main__":
#     pdf_file = "mercedes.pdf"     # ← твой PDF
#     ocr_pdf_text_only(pdf_file, "mercedes_clean.txt")
