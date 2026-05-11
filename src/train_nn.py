import argparse
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_FRACTION_OF_TRAIN = 0.20
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

JOB_SPECS = (
    ("cnn", "iq"),
    ("mlp", "bins"),
    ("mlp", "feat"),
    ("mlp", "spec"),
    ("mlp", "iq_flat"),
)

# Architectures
class ConvNet1D(nn.Module):
    # CNN for raw I/Q.

    def __init__(self, n_input_samples=1600):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(64 * 8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


class DeepMLP(nn.Module):
    # MLP
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x).squeeze(-1)

# Data
def load_dataset(h5_path, group):
    if group == "spec":
        with h5py.File(h5_path, "r") as f:
            X_iq = f["iq/data"][:].T
            a3 = f["iq/a3"][:].flatten()
            y = f["iq/nonlinear"][:].flatten().astype(int)
        n = X_iq.shape[1] // 2
        complex_iq = X_iq[:, :n] + 1j * X_iq[:, n:]
        X = np.abs(np.fft.fft(complex_iq, axis=-1)) ** 2
        return X.astype(np.float32), a3, y

    if group == "iq_flat":
        with h5py.File(h5_path, "r") as f:
            X = f["iq/data"][:].T
            a3 = f["iq/a3"][:].flatten()
            y = f["iq/nonlinear"][:].flatten().astype(int)
        return X.astype(np.float32), a3, y

    with h5py.File(h5_path, "r") as f:
        X = f[f"{group}/data"][:].T
        a3 = f[f"{group}/a3"][:].flatten()
        y = f[f"{group}/nonlinear"][:].flatten().astype(int)
    if group == "iq":
        # Create (2, 16000) shape for CNN
        n = X.shape[1] // 2
        X = np.stack([X[:, :n], X[:, n:]], axis=1)
    return X.astype(np.float32), a3, y


def split_per_a3(X, y, a3_array, a3_value):
    mask = a3_array == a3_value
    X_a, y_a = X[mask], y[mask]
    X_trval, X_test, y_trval, y_test = train_test_split(
        X_a,
        y_a,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_a,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval,
        y_trval,
        test_size=VAL_FRACTION_OF_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_trval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize(X_train, X_val, X_test, is_iq):
    if is_iq:
        train_shape = X_train.shape
        val_shape = X_val.shape
        test_shape = X_test.shape
        X_train = X_train.reshape(train_shape[0], -1)
        X_val = X_val.reshape(val_shape[0], -1)
        X_test = X_test.reshape(test_shape[0], -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if is_iq:
        X_train = X_train.reshape(train_shape)
        X_val = X_val.reshape(val_shape)
        X_test = X_test.reshape(test_shape)
    return X_train, X_val, X_test


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
    gen = torch.Generator()
    gen.manual_seed(RANDOM_STATE)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)

# Training
def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    preds, scores, ys = [], [], []
    n_total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            losses.append(loss.item() * y.size(0))
            n_total += y.size(0)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            preds.append((logits > 0).long().cpu().numpy())
            ys.append(y.long().cpu().numpy())
    avg_loss = sum(losses) / n_total
    y_arr = np.concatenate(ys)
    pred_arr = np.concatenate(preds)
    score_arr = np.concatenate(scores)
    return avg_loss, accuracy_score(y_arr, pred_arr), y_arr, pred_arr, score_arr


def train_one_fit(arch_name, X_train, y_train, X_val, y_val, X_test, y_test, device):
    if arch_name == "cnn":
        model = ConvNet1D(n_input_samples=X_train.shape[-1]).to(device)
    else:
        model = DeepMLP(in_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        correct = 0
        n_total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += ((logits > 0).long() == y.long()).sum().item()
            n_total += y.size(0)
        train_loss = loss_sum / n_total
        train_acc = correct / n_total

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    train_time = time.time() - t0

    model.load_state_dict(best_state)
    _, test_acc, y_t, y_p, y_s = evaluate(model, test_loader, device, criterion)
    test_auc = roc_auc_score(y_t, y_s)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()

    metrics = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "test_accuracy": float(test_acc),
        "auc": float(test_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_time_sec": float(train_time),
    }
    return metrics, history, y_t, y_p, y_s

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    set_seed(RANDOM_STATE)

    h5_in = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    preds_path = out_dir / "predictions.h5"
    if preds_path.exists():
        preds_path.unlink()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    rows = []
    with h5py.File(preds_path, "w") as preds_f:
        for arch, dset in JOB_SPECS:
            print(f"\n=== {arch} on /{dset} ===", flush=True)
            X_full, a3, y = load_dataset(h5_in, dset)
            a3_unique = np.sort(np.unique(a3))
            print(f"  X shape: {X_full.shape}, a3 values: {a3_unique}", flush=True)

            for a3_idx, a3_val in enumerate(a3_unique):
                X_train, X_val, X_test, y_train, y_val, y_test = split_per_a3(
                    X_full, y, a3, a3_val
                )
                X_train, X_val, X_test = normalize(
                    X_train, X_val, X_test, is_iq=(dset == "iq")
                )
                print(
                    f"  fit {arch}/{dset} a3={a3_val:.4f} "
                    f"(train={len(X_train)}, val={len(X_val)}, test={len(X_test)})",
                    flush=True,
                )
                metrics, history, y_t, y_p, y_s = train_one_fit(
                    arch, X_train, y_train, X_val, y_val, X_test, y_test, device
                )
                print(
                    f"    best_val_acc={metrics['best_val_acc']:.4f} "
                    f"(epoch {metrics['best_epoch']})  "
                    f"test_acc={metrics['test_accuracy']:.4f}  "
                    f"auc={metrics['auc']:.4f}  "
                    f"time={metrics['train_time_sec']:.1f}s",
                    flush=True,
                )

                rows.append(
                    {
                        "architecture": arch,
                        "dataset": dset,
                        "a3": float(a3_val),
                        "a3_idx": int(a3_idx),
                        "n_train": len(X_train),
                        "n_val": len(X_val),
                        "n_test": len(X_test),
                        **metrics,
                    }
                )

                grp = preds_f.create_group(f"{arch}/{dset}/a3_{a3_idx}")
                grp.attrs["a3_value"] = float(a3_val)
                grp.create_dataset("y_test", data=y_t.astype(np.uint8))
                grp.create_dataset("y_pred", data=y_p.astype(np.uint8))
                grp.create_dataset("y_score", data=y_s.astype(np.float64))
                grp.create_dataset(
                    "train_loss", data=np.array(history["train_loss"], dtype=np.float64)
                )
                grp.create_dataset(
                    "val_loss", data=np.array(history["val_loss"], dtype=np.float64)
                )
                grp.create_dataset(
                    "train_acc", data=np.array(history["train_acc"], dtype=np.float64)
                )
                grp.create_dataset(
                    "val_acc", data=np.array(history["val_acc"], dtype=np.float64)
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "architecture",
            "dataset",
            "a3",
            "a3_idx",
            "n_train",
            "n_val",
            "n_test",
            "best_val_acc",
            "best_epoch",
            "test_accuracy",
            "auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "train_time_sec",
        ],
    )
    df.to_csv(metrics_path, index=False)
    print(f"\nWrote {metrics_path} ({len(df)} rows)", flush=True)
    print(f"Wrote {preds_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        default="/work/pi_mduarte_umass_edu/oraftery_umass_edu/data/simulation_data.h5",
    )
    p.add_argument(
        "--out-dir",
        default="/work/pi_mduarte_umass_edu/oraftery_umass_edu/results/nn",
    )
    p.add_argument(
        "--cpu", action="store_true"
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
