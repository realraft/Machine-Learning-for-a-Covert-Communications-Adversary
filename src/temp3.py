import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "C:\\Users\\oaraf\\Desktop\\Honors Thesis\\data\\data.csv"

df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
y = df['nonlinear'].to_numpy(dtype=int)

print(df[df.nonlinear==0].head())
print(df[df.nonlinear==1].head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
lr = LogisticRegression(max_iter=1000)
lr.fit(scaler.fit_transform(X_train), y_train)

print("LR coefficients:")
for name, coef in zip(df.columns[:-1], lr.coef_[0]):
    print(f"  {name:25s}: {coef:+.4f}")

# Check if LR is just memorizing — test on completely held-out a3 values
# What's the decision boundary value for each sample?
scores = lr.decision_function(scaler.transform(X))
print(f"\nDecision score range: [{scores.min():.2f}, {scores.max():.2f}]")
print(f"Scores near zero (ambiguous): {(np.abs(scores) < 0.1).sum()}")
print(f"Scores > 10 (very confident): {(np.abs(scores) > 10).sum()}")
print(f"Scores > 100 (absurdly confident): {(np.abs(scores) > 100).sum()}")