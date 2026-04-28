import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Constants
CSV_PATH = "/work/pi_mduarte_umass_edu/oraftery_umass_edu/data/data.csv"
MODEL_SAVE_PATH = "/work/pi_mduarte_umass_edu/oraftery_umass_edu/models/final_dnn.csv"
LABEL_COL = "nonlinear"
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Load pre-engineered features data
# For local testing, fallback to relative path if absolute doesn't exist
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    df = pd.read_csv("../data/data.csv")

assert LABEL_COL in df.columns, f"Missing label column: {LABEL_COL}"
assert np.isin(df[LABEL_COL], [0, 1]).all(), "Labels must be binary: 0 (linear), 1 (nonlinear)."

y = df[LABEL_COL].to_numpy(dtype=int)

class_counts = pd.Series(y, name=LABEL_COL).value_counts().sort_index()
print("Class distribution (0=linear, 1=nonlinear):")
print(class_counts)
print(f"\nDataset shape: {df.shape}")
print(f"Feature columns: {list(df.columns[:-1])}")

# Display feature summary statistics by class
features_df = df.drop(columns=[LABEL_COL])

print("\nMean feature value by class (0=linear, 1=nonlinear):")
print(features_df.assign(nonlinear=y).groupby("nonlinear").mean())

print("\nStandard deviation by class:")
print(features_df.assign(nonlinear=y).groupby("nonlinear").std())

# Prepare features and train/test split
X = df.drop(columns=[LABEL_COL])
y_target = y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_target,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_target,
)

# Scale data without leakage: fit on train only, transform train/test with same scaler
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

# Initialize model results table
results_df = pd.DataFrame(
    columns=[
        "model",
        "train_accuracy",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "auc_mean",
    ]
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train class balance: {np.bincount(y_train)}")
print(f"Test class balance: {np.bincount(y_test)}")

# -----------------
# PyTorch DNN model
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Prepare DataLoaders
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss, optimizer
model = SimpleDNN(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.apply_step() if hasattr(optimizer, 'apply_step') else optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
            predictions = (outputs >= 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            
    val_loss /= len(test_loader.dataset)
    val_accuracy = correct / total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

# Final Evaluation
model.eval()
all_preds = []
all_probs = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        probs = model(batch_X)
        preds = (probs >= 0.5).float()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_probs = np.array(all_probs).flatten()

test_acc = accuracy_score(y_test, all_preds)
test_prec = precision_score(y_test, all_preds)
test_rec = recall_score(y_test, all_preds)
test_f1 = f1_score(y_test, all_preds)
test_auc = roc_auc_score(y_test, all_probs)

print("\n--- Final Test Metrics ---")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"F1-Score : {test_f1:.4f}")
print(f"AUC      : {test_auc:.4f}")

# Save the final model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel state dict saved to: {MODEL_SAVE_PATH}")
