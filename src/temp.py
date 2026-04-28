import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr

# =========================================================================
# CONFIGURATION
# =========================================================================

CSV_PATH = "C:\\Users\\oaraf\\Desktop\\Honors Thesis\\data\\data.csv"

INPUT_SIZE = 7  # you have 7 features, not 10
RANDOM_STATE = 42

# =========================================================================
# PART 1: LOAD AND ANALYZE DATA
# =========================================================================

print("\n=== BLIND FEATURE ANALYSIS ===\n", flush=True)

df = pd.read_csv(CSV_PATH)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}", flush=True)
print(f"Class distribution:\n{df['nonlinear'].value_counts()}\n", flush=True)

X = df.drop(columns='nonlinear').to_numpy(dtype=np.float32)
y = df['nonlinear'].to_numpy(dtype=int)

# Feature correlation analysis
print("Feature correlations with nonlinearity (Willie's perspective):", flush=True)
print("(These are blind features extracted without protocol knowledge)\n", flush=True)

for col in df.columns[:-1]:
    corr, pval = pointbiserialr(y, df[col])
    marker = "***" if pval < 1e-10 else ("**" if pval < 0.001 else ("*" if pval < 0.05 else ""))
    print(f"  {col:25s}: corr={corr:7.4f}, pval={pval:.2e} {marker}", flush=True)

print("\n*** p<1e-10, ** p<0.001, * p<0.05\n", flush=True)

# =========================================================================
# PART 2: BASELINE - LOGISTIC REGRESSION
# =========================================================================

print("=== BASELINE: LOGISTIC REGRESSION ===\n", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_train_acc = lr.score(X_train_scaled, y_train)
lr_test_acc = lr.score(X_test_scaled, y_test)
lr_test_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

print(f"LogReg Train Accuracy: {lr_train_acc:.4f}", flush=True)
print(f"LogReg Test Accuracy:  {lr_test_acc:.4f}", flush=True)
print(f"LogReg Test AUC:       {lr_test_auc:.4f}", flush=True)
print(f"\nInterpretation: Linear boundaries give {lr_test_acc:.1%} accuracy.\n", flush=True)