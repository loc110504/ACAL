import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np

# Load file với tất cả các cột
df = pd.read_csv("answers/hearsay/mad_gemini_hearsay.tsv", sep='\t')

# Kiểm tra các cột có trong file
print("Columns in file:", df.columns.tolist())
print(f"\nTotal rows: {len(df)}")
print(f"First few rows:\n{df.head()}\n")

# Chuẩn hóa các giá trị Yes/No (uppercase)
df['gold_answer'] = df['gold_answer'].str.strip().str.capitalize()
df['llm_answer'] = df['llm_answer'].str.strip().str.capitalize()

# Xử lý các giá trị null trong llm_answer
null_count = df['llm_answer'].isna().sum()
if null_count > 0:
    print(f"⚠️ Warning: {null_count} samples have null llm_answer (will be excluded from metrics)\n")
    df_valid = df.dropna(subset=['llm_answer'])
else:
    df_valid = df

# Lấy cột gold_answer và llm_answer
y_true = df_valid['gold_answer']
y_pred = df_valid['llm_answer']

# Tính các metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Tính metrics cho từng class
precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=['Yes', 'No'])
recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=['Yes', 'No'])
f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=['Yes', 'No'])

# In kết quả tổng quan
print("=" * 60)
print("CLASSIFICATION METRICS (Macro Average)")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print("=" * 60)

# In metrics cho từng class
print("\nPer-Class Metrics:")
print("-" * 60)
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)
print(f"{'Yes':<10} {precision_per_class[0]:<12.4f} {recall_per_class[0]:<12.4f} {f1_per_class[0]:<12.4f}")
print(f"{'No':<10} {precision_per_class[1]:<12.4f} {recall_per_class[1]:<12.4f} {f1_per_class[1]:<12.4f}")
print("-" * 60)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=['Yes', 'No'])
print(f"{'':>12} {'Pred Yes':>10} {'Pred No':>10}")
print(f"{'True Yes':<12} {cm[0][0]:>10} {cm[0][1]:>10}")
print(f"{'True No':<12} {cm[1][0]:>10} {cm[1][1]:>10}")

# In classification report chi tiết
print("\n" + "=" * 60)
print("Detailed Classification Report:")
print("=" * 60)
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# Thống kê chi tiết
print("=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Total samples in file:     {len(df)}")
print(f"Valid samples (non-null):  {len(df_valid)}")
print(f"Failed samples (null):     {null_count}")
print()
print(f"Gold 'Yes':                {(y_true == 'Yes').sum()}")
print(f"Gold 'No':                 {(y_true == 'No').sum()}")
print(f"Predicted 'Yes':           {(y_pred == 'Yes').sum()}")
print(f"Predicted 'No':            {(y_pred == 'No').sum()}")
print()
print(f"Correct predictions:       {(y_true == y_pred).sum()}")
print(f"Wrong predictions:         {(y_true != y_pred).sum()}")