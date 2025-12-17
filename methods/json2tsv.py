import json
import csv

# Đường dẫn file
input_json = "answers/learned_hands_courts/mad_gemini_courts.json"
output_tsv = "answers/learned_hands_courts/mad_gemini_courts.tsv"

# Đọc JSON
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Các cột cần giữ
fields = ["index", "text", "gold_answer", "final_answer"]

# Ghi TSV
with open(output_tsv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        delimiter="\t",
        quoting=csv.QUOTE_MINIMAL
    )
    writer.writeheader()

    for item in data:
        row = {field: item.get(field, "") for field in fields}
        writer.writerow(row)

print(f"Đã chuyển xong → {output_tsv}")
