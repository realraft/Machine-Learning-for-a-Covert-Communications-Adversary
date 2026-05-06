"""
DNN training pipeline for binary classification of linear vs nonlinear RF signals.

Architecture matches Youssef et al. 2018 (IEEE JRFID) DNN baseline:
  - Two fully-connected hidden layers, 128 nodes each
  - ReLU hidden activations
  - Adam optimizer, mini-batch size 32
  - Standard-scaled features

Assumptions:
  - Data is clean and classes are balanced
  - Label column is named 'nonlinear' or is the last column
  - Input features are concatenated [Re, Im] IQ samples
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)

    if 'nonlinear' in df.columns:
        y = df['nonlinear'].astype(int).values
        X = df.drop(columns=['nonlinear']).values.astype(np.float32)
    else:
        y = df.iloc[:, -1].astype(int).values
        X = df.iloc[:, :-1].values.astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def data_sanity_check(X, y):
    """
    Print per-class power and inter-class feature distance.
    If power differs between classes, the power test has not been blinded.
    If inter-class diff is near zero, the simulation produced identical classes.
    """
    half = X.shape[1] // 2
    X_re = X[:, :half]
    X_im = X[:, half:]

    print("--- Data Sanity Checks ---")
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    total  = sum(counts.values())
    for cls in sorted(counts):
        mask = y == cls
        pwr  = np.mean(X_re[mask] ** 2 + X_im[mask] ** 2)
        print(f"  Class {cls}  n={counts[cls]:5d} ({100*counts[cls]/total:.1f}%)  "
              f"mean power: {pwr:.6f}")

    majority_baseline = max(counts.values()) / total
    print(f"  Majority-class baseline accuracy : {majority_baseline:.4f}")

    classes = sorted(counts)
    if len(classes) == 2:
        ratio = max(counts.values()) / min(counts.values())
        if ratio > 1.5:
            print(f"  WARNING: imbalance ratio {ratio:.1f}x -- minority class will be "
                  f"oversampled during training.")

    inter_class_diff = np.mean(
        np.abs(X[y == classes[0]].mean(axis=0) - X[y == classes[-1]].mean(axis=0))
    )
    print(f"  Inter-class mean-feature diff    : {inter_class_diff:.6f}")
    if inter_class_diff < 1e-4:
        print("  WARNING: classes look nearly identical -- check simulation noise.")
    print("--------------------------\n")


# ---------------------------------------------------------------------------
# Oversampling (MLPClassifier has no class_weight argument)
# ---------------------------------------------------------------------------

def oversample_minority(X_train, y_train, random_state=42):
    """
    Duplicate minority-class rows until classes are balanced.
    Applied only to the training set so val/test metrics stay honest.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    if counts.max() == counts.min():
        return X_train, y_train

    rng      = np.random.default_rng(random_state)
    majority = classes[counts.argmax()]
    n_target = counts.max()

    X_parts, y_parts = [X_train], [y_train]
    for cls, n in zip(classes, counts):
        if cls == majority:
            continue
        deficit = n_target - n
        idx     = rng.choice(np.where(y_train == cls)[0], size=deficit, replace=True)
        X_parts.append(X_train[idx])
        y_parts.append(y_train[idx])

    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)

    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


# ---------------------------------------------------------------------------
# Classical detectors (baseline -- should fail when power is equalized)
# ---------------------------------------------------------------------------

def classical_detectors(X_re, X_im):
    x_complex = X_re + 1j * X_im

    power    = np.mean(np.abs(x_complex) ** 2, axis=1)
    kurtosis = stats.kurtosis(np.abs(x_complex), axis=1)
    papr     = (np.max(np.abs(x_complex) ** 2, axis=1)
                / np.mean(np.abs(x_complex) ** 2, axis=1))
    autocorr = np.array([
        np.abs(np.corrcoef(r[:-2], r[2:])[0, 1])
        for r in np.abs(x_complex)
    ])

    return {
        'power'   : power,
        'kurtosis': kurtosis,
        'papr'    : papr,
        'autocorr': autocorr,
    }


def threshold_accuracy(feature, labels, n_thresholds=100):
    """Best accuracy achievable by a single threshold on this feature."""
    thresholds = np.linspace(feature.min(), feature.max(), n_thresholds)
    return max(
        max(np.mean((feature > t) == labels),
            np.mean((feature < t) == labels))
        for t in thresholds
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class DNNSurrogate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.net(x)


def main(args):
    X, y = load_data(args.data)
    print(f"Loaded data  X={X.shape}  y={y.shape}")
    print(f"Class distribution: { {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))} }\n")

    # Sanity checks
    data_sanity_check(X, y)

    # Classical detectors (full dataset -- upper-bound reference)
    half = X.shape[1] // 2
    X_re = X[:, :half]
    X_im = X[:, half:]

    print("--- Classical Detectors Baseline (best single threshold, full dataset) ---")
    for name, feat in classical_detectors(X_re, X_im).items():
        acc = threshold_accuracy(feat, y)
        print(f"  {name.capitalize():<12}: {acc:.4f}")
    print("  Note: computed on all samples -- treat as an upper-bound reference.")
    print("  The DNN sees only the training split, making this comparison strict.")
    print("--------------------------------------------------------------------------\n")

    # Train / val / test split (70 / 15 / 15)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print(f"Split  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # Oversample minority class in training set only
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train)
    if len(y_train_bal) != len(y_train):
        added = len(y_train_bal) - len(y_train)
        print(f"Oversampled training set: +{added} minority rows -> {len(y_train_bal)} total")
    print()

    # Feature scaling (fit on balanced training set only)
    scaler      = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_val       = scaler.transform(X_val)
    X_test      = scaler.transform(X_test)

    # PyTorch DNN matching Youssef et al. 2018:
    #   Two hidden layers of 128 nodes, ReLU, Adam, batch size 32.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build datasets and dataloaders
    X_train_t = torch.tensor(X_train_bal, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_bal, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = X_train_bal.shape[1]
    model = DNNSurrogate(input_dim).to(device)
    
    # alpha=1e-3 in MLPClassifier corresponds to L2 penalty -> weight_decay=1e-3
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    print("Training DNN ...")
    best_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = args.epochs
    patience = 50
    tol = 1e-6
    n_iter_ = 0
    best_model_state = model.state_dict()

    for epoch in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t.to(device))
            val_loss = criterion(val_out, y_val_t.to(device)).item()
            
        n_iter_ += 1
        if best_loss - val_loss > tol:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    print(f"Stopped after {n_iter_} iterations\n")

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_preds = (torch.sigmoid(model(X_val_t.to(device))).cpu().numpy() > 0.5).astype(int).flatten()
        test_preds = (torch.sigmoid(model(X_test_t.to(device))).cpu().numpy() > 0.5).astype(int).flatten()

    majority_baseline = max(np.bincount(y_test)) / len(y_test)
    val_acc           = accuracy_score(y_val, val_preds)
    test_acc          = accuracy_score(y_test, test_preds)

    print(f"Majority-class baseline : {majority_baseline:.4f}")
    print(f"Val  accuracy           : {val_acc:.4f}")
    print(f"Test accuracy           : {test_acc:.4f}\n")
    print(classification_report(y_test, test_preds))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=str, default='/work/pi_mduarte_umass_edu/oraftery_umass_edu/data/data.csv')
    p.add_argument('--epochs',     type=int, default=500)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()
    main(args)
