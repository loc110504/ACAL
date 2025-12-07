import pandas as pd
import ast
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load file
df = pd.read_csv("standard_prompt_phi4_hearsay.tsv", sep="\t")

# Hàm parse chuỗi trả về dạng "{'answer': 'No', 'explanation': '...'}"
def parse_llm_output(text):
    try:
        # ast.literal_eval xử lý Python-dict string an toàn hơn json.loads
        data = ast.literal_eval(text)
        return data.get("answer", None)
    except Exception:
        return None

# Tạo cột prediction
df["pred_answer"] = df["llm_output"].apply(parse_llm_output)

# Normalize label (Yes/No)
df["pred_answer"] = df["pred_answer"].astype(str).str.strip().str.capitalize()
df["gold_answer"] = df["gold_answer"].astype(str).str.strip().str.capitalize()

# Evaluation
y_true = df["gold_answer"]
y_pred = df["pred_answer"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Macro:", f1_score(y_true, y_pred, average="macro"))
print("F1 Micro:", f1_score(y_true, y_pred, average="micro"))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
