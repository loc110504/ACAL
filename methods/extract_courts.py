
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load file
df = pd.read_csv("answers/learned_hands_courts/cot_prompt_llama_courts.tsv",  sep='\t', usecols=[0, 1, 2, 3, 4], 
                 names=['index', 'text', 'gold_answer', 'llm_answer', 'llm_explanation'],
                 skiprows=1)

# Lấy cột gold_answer và llm_answer
y_true = df['gold_answer']
y_pred = df['llm_answer']

# Tính các metrics (bỏ pos_label khi dùng average='macro')
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

# In kết quả
print("=" * 50)
print("CLASSIFICATION METRICS (Macro Average)")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print("=" * 50)

# In classification report chi tiết
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Thống kê thêm
print(f"\nTotal samples: {len(df)}")
print(f"Gold 'Yes': {(y_true == 'Yes').sum()}")
print(f"Gold 'No': {(y_true == 'No').sum()}")
print(f"Predicted 'Yes': {(y_pred == 'Yes').sum()}")
print(f"Predicted 'No': {(y_pred == 'No').sum()}")