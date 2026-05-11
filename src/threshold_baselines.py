"""Threshold-test detector baselines for the covert-comms adversary thesis.

For each (dataset, statistic, a3) combination, compute a single scalar
test statistic per sample, then sweep the threshold to find the best
achievable test accuracy. Reports per-fit accuracy, AUC, and writes the
full per-sample scores so the plotting script can derive ROC curves.

Datasets:
  /bins -- matched-filter PSD vector (M x N real)
  /iq   -- raw I/Q samples (M x 2N real, [Re | Im])
Note: the engineered-feature dataset (/feat) is excluded from threshold
tests since the engineered features ARE essentially threshold statistics
themselves.

Statistics (each computed per sample):
  power     -- mean |x|^2 (time domain) / mean of bins (spectrum)
  kurtosis  -- excess kurtosis of |x| / of bin values
  papr      -- max(|x|^2)/mean(|x|^2) / max(bin)/mean(bin)
  autocorr  -- |corr(r[:-2], r[2:])| of |x| or of bin sequence

Same train/test split as the ML and NN pipelines (70/30, seed 42,
stratified by class) -- only the test rows are used. Training data is
not needed since these detectors have no parameters to fit.

Sign convention: if a statistic is anti-correlated with class 1 (AUC
< 0.5), its score is negated so that the saved scores always satisfy
"higher score => more likely class 1" -- mirrors the classical_ml
predictions.h5 layout exactly.

Outputs:
  results/threshold_baselines/metrics.csv     -- one row per (dataset, statistic, a3)
  results/threshold_baselines/predictions.h5  -- y_test, y_pred, y_score per fit
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.30
DATASETS = ("bins", "iq")
STATISTICS = ("power", "kurtosis", "papr", "autocorr", "regrowth_band_power")

# SRRC pulse parameters (same as MATLAB simulation)
BETA = 0.25
OSR = 16
MAIN_BAND_EDGE = (1 + BETA) / 2  # cycles / symbol period


def load_dataset_flat(h5_path, group):
    with h5py.File(h5_path, "r") as f:
        X = f[f"{group}/data"][:].T
        a3 = f[f"{group}/a3"][:].flatten()
        y = f[f"{group}/nonlinear"][:].flatten().astype(int)
    return X, a3, y


def lag2_autocorr(rows):
    out = np.empty(rows.shape[0])
    for i, r in enumerate(rows):
        c = np.corrcoef(r[:-2], r[2:])[0, 1]
        out[i] = np.abs(c) if np.isfinite(c) else 0.0
    return out


def stop_band_mask(n_bins):
    #Mask to select bins outside main band.
    k = np.arange(n_bins)
    f = np.where(k < n_bins / 2, k * OSR / n_bins, (k - n_bins) * OSR / n_bins)
    return np.abs(f) > MAIN_BAND_EDGE


def stats_iq(X_iq_flat, statistic):
    """Calculate statistics on /iq dataset"""
    n = X_iq_flat.shape[1] // 2
    x_complex = X_iq_flat[:, :n] + 1j * X_iq_flat[:, n:]
    mag = np.abs(x_complex)
    if statistic == "power":
        return np.mean(mag**2, axis=1)
    if statistic == "kurtosis":
        return stats.kurtosis(mag, axis=1)
    if statistic == "papr":
        return np.max(mag**2, axis=1) / np.mean(mag**2, axis=1)
    if statistic == "autocorr":
        return lag2_autocorr(mag)
    if statistic == "regrowth_band_power":
        spec = np.abs(np.fft.fft(x_complex, axis=1)) ** 2
        mask = stop_band_mask(spec.shape[1])
        return spec[:, mask].sum(axis=1)
    raise ValueError(f"Unknown statistic: {statistic}")


def stats_bins(X_bins, statistic):
    """Calculate statistics on /bins dataset"""
    if statistic == "power":
        return np.mean(X_bins, axis=1)
    if statistic == "kurtosis":
        return stats.kurtosis(X_bins, axis=1)
    if statistic == "papr":
        return np.max(X_bins, axis=1) / np.mean(X_bins, axis=1)
    if statistic == "autocorr":
        return lag2_autocorr(X_bins)
    if statistic == "regrowth_band_power":
        mask = stop_band_mask(X_bins.shape[1])
        return X_bins[:, mask].sum(axis=1)
    raise ValueError(f"Unknown statistic: {statistic}")


def best_threshold_accuracy(scores, y):
    # Sweep all decision boundaries on scores and return the threshold maximizing accuracy
    unique = np.unique(scores)
    if len(unique) == 1:
        # If all scores equal, pick majority class
        majority_acc = max(np.mean(y == 0), np.mean(y == 1))
        return float(unique[0]), float(majority_acc)
    midpoints = (unique[:-1] + unique[1:]) / 2
    candidates = np.concatenate([[-np.inf], midpoints, [np.inf]])
    accuracies = np.array([np.mean((scores >= t) == y) for t in candidates])
    best_idx = int(np.argmax(accuracies))
    return float(candidates[best_idx]), float(accuracies[best_idx])


def run(args):
    h5_in = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    preds_path = out_dir / "predictions.h5"
    if preds_path.exists():
        preds_path.unlink()

    rows = []
    with h5py.File(preds_path, "w") as preds_f:
        for dset in DATASETS:
            print(f"\n=== /{dset} ===", flush=True)
            X, a3, y = load_dataset_flat(h5_in, dset)
            a3_unique = np.sort(np.unique(a3))

            for a3_idx, a3_val in enumerate(a3_unique):
                mask = a3 == a3_val
                X_a, y_a = X[mask], y[mask]
                _, X_test, _, y_test = train_test_split(
                    X_a,
                    y_a,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_a,
                )

                for stat_name in STATISTICS:
                    scores = (
                        stats_iq(X_test, stat_name)
                        if dset == "iq"
                        else stats_bins(X_test, stat_name)
                    )
                    auc = roc_auc_score(y_test, scores)
                    if auc < 0.5:
                        scores = -scores
                        auc = 1.0 - auc
                    best_t, best_acc = best_threshold_accuracy(scores, y_test)
                    y_pred = (scores >= best_t).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                    print(
                        f"  {dset}/{stat_name} a3={a3_val:.4f}: "
                        f"best_acc={best_acc:.4f}  auc={auc:.4f}",
                        flush=True,
                    )

                    rows.append(
                        {
                            "dataset": dset,
                            "statistic": stat_name,
                            "a3": float(a3_val),
                            "a3_idx": int(a3_idx),
                            "n_test": int(len(y_test)),
                            "best_threshold": float(best_t),
                            "best_accuracy": float(best_acc),
                            "auc": float(auc),
                            "tn": int(tn),
                            "fp": int(fp),
                            "fn": int(fn),
                            "tp": int(tp),
                        }
                    )

                    grp = preds_f.create_group(f"{dset}/{stat_name}/a3_{a3_idx}")
                    grp.attrs["a3_value"] = float(a3_val)
                    grp.attrs["best_threshold"] = float(best_t)
                    grp.create_dataset("y_test", data=y_test.astype(np.uint8))
                    grp.create_dataset("y_pred", data=y_pred.astype(np.uint8))
                    grp.create_dataset("y_score", data=scores.astype(np.float64))

    df = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "statistic",
            "a3",
            "a3_idx",
            "n_test",
            "best_threshold",
            "best_accuracy",
            "auc",
            "tn",
            "fp",
            "fn",
            "tp",
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
        default="/work/pi_mduarte_umass_edu/oraftery_umass_edu/results/threshold_baselines",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
