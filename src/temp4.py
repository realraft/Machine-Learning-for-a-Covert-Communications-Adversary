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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

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
    If power differs between classes, the power test hasn't been blinded.
    If inter-class diff is near zero, the simulation produced identical classes.
    """
    half = X.shape[1] // 2
    X_re = X[:, :half]
    X_im = X[:, half:]

    print("--- Data Sanity Checks ---")
    for cls in sorted(np.unique(y)):
        mask = y == cls
        pwr = np.mean(X_re[mask] ** 2 + X_im[mask] ** 2)
        print(f"  Class {cls} mean power : {pwr:.6f}")

    inter_class_diff = np.mean(
        np.abs(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0))
    )
    print(f"  Inter-class mean-feature diff : {inter_class_diff:.6f}")
    if inter_class_diff < 1e-4:
        print("  WARNING: classes look nearly identical — check simulation noise.")
    print("--------------------------\n")


# ---------------------------------------------------------------------------
# Classical detectors (baseline — should fail when power is equalized)
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

def main(args):
    X, y = load_data(args.data)
    print(f"Loaded data  X={X.shape}  y={y.shape}")
    print(f"Class distribution: { {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))} }\n")

    # ---- Sanity checks ----
    data_sanity_check(X, y)

    # ---- Classical detectors ----
    half = X.shape[1] // 2
    X_re = X[:, :half]
    X_im = X[:, half:]

    print("--- Classical Detectors Baseline (best single threshold) ---")
    for name, feat in classical_detectors(X_re, X_im).items():
        acc = threshold_accuracy(feat, y)
        print(f"  {name.capitalize():<12}: {acc:.4f}")
    print("------------------------------------------------------------\n")

    # ---- Train / val / test split (70 / 15 / 15) ----
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print(f"Split  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}\n")

    # ---- Feature scaling ----
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ---- DNN — Youssef et al. 2018 architecture ----
    #   Two hidden layers, 128 nodes each, ReLU, Adam, batch size 32.
    #   alpha adds L2 regularization (helps with limited data).
    model = MLPClassifier(
        hidden_layer_sizes=(128, 128),   # paper: two layers of 128 nodes
        activation='relu',
        solver='adam',
        alpha=1e-3,                      # L2 regularization
        batch_size=32,                   # paper: mini-batch 32
        max_iter=args.epochs,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )

    print("Training DNN …")
    model.fit(X_train, y_train)
    print(f"Stopped after {model.n_iter_} iterations\n")

    # ---- Evaluate ----
    val_acc  = accuracy_score(y_val,  model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Val  accuracy : {val_acc:.4f}")
    print(f"Test accuracy : {test_acc:.4f}\n")
    print(classification_report(y_test, model.predict(X_test)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=str, default='data/data.csv')
    p.add_argument('--epochs',     type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()
    main(args)