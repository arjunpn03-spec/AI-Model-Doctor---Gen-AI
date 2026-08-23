import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

print("==========================================")
print("🩺 AI MODEL DOCTOR — SURGICAL TREATMENT (SMOTE)")
print("==========================================")

# 1. Load and Split the Data
df = pd.read_csv('data/churn_data.csv')
X = df.drop(columns=['Churn'])
y = df['Churn']

# We only split here, we DO NOT apply SMOTE to the test data. 
# The test data must remain a realistic reflection of the real world!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Train the Baseline Model (The "Sick" Patient)
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
base_preds = base_model.predict(X_test)

base_rec = recall_score(y_test, base_preds)
base_f1 = f1_score(y_test, base_preds)
base_acc = accuracy_score(y_test, base_preds)

# 3. Apply Data Surgery: SMOTE
print("💉 Performing SMOTE Data Surgery (Synthesizing new minority data)...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"   Original training churn count: {sum(y_train == 1)}")
print(f"   Post-surgery training churn count: {sum(y_train_smote == 1)}\n")

# 4. Train the Treated Model
print("🧠 Retraining model on synthesized dataset...")
treated_model = RandomForestClassifier(random_state=42)
treated_model.fit(X_train_smote, y_train_smote)
treated_preds = treated_model.predict(X_test)

treated_rec = recall_score(y_test, treated_preds)
treated_f1 = f1_score(y_test, treated_preds)
treated_acc = accuracy_score(y_test, treated_preds)

# 5. Display the Results (Before vs After)
def get_trend(before, after):
    if after > before + 0.005: return "↑"
    if after < before - 0.005: return "↓"
    return "→"

print("==========================================")
print("📋 POST-TREATMENT VITALS (BEFORE vs AFTER)")
print("==========================================")
print(f"{'Metric':<12} | {'Before':<8} | {'After':<8} | {'Trend'}")
print("-" * 45)
print(f"{'Recall':<12} | {base_rec:.2%}   | {treated_rec:.2%}   |   {get_trend(base_rec, treated_rec)}")
print(f"{'F1 Score':<12} | {base_f1:.2%}   | {treated_f1:.2%}   |   {get_trend(base_f1, treated_f1)}")
print(f"{'Accuracy':<12} | {base_acc:.2%}   | {treated_acc:.2%}   |   {get_trend(base_acc, treated_acc)}")
print("==========================================\n")