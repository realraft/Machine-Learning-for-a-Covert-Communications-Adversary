import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

RANDOM_STATE = 42
TEST_SIZE = 0.30
DATASETS = ("bins", "feat", "iq")
CLASSIFIER_ORDER = ("logreg", "svm", "rf", "gb")


def n_cpus_available():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def load_dataset(h5_path, group):
    with h5py.File(h5_path, "r") as f:
        X = f[f"{group}/data"][:].T
        a3 = f[f"{group}/a3"][:].flatten()
        y = f[f"{group}/nonlinear"][:].flatten().astype(int)
    return X, a3, y


def make_classifiers(dataset_name, n_jobs):
    if dataset_name == "iq":
        svm = LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=RANDOM_STATE)
    else:
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    return {
        "logreg": LogisticRegression(
            max_iter=1000, n_jobs=n_jobs, random_state=RANDOM_STATE
        ),
        "svm": svm,
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
        ),
        "gb": HistGradientBoostingClassifier(
            max_depth=5,
            max_iter=200,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def get_score(model, X):
    # Return decision scores for class 1
    if isinstance(model, (SVC, LinearSVC)):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def evaluate_fit(model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    y_score = get_score(model, X_test)
    test_acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return (
        {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "auc": float(auc),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        y_pred,
        y_score,
    )


def run(args):
    h5_in = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    preds_path = out_dir / "predictions.h5"
    if preds_path.exists():
        preds_path.unlink()

    n_jobs = n_cpus_available()
    print(f"Using {n_jobs} CPU(s).", flush=True)

    rows = []
    with h5py.File(preds_path, "w") as preds_f:
        for dset in DATASETS:
            print(f"\n=== /{dset} ===", flush=True)
            X, a3, y = load_dataset(h5_in, dset)
            a3_unique = np.sort(np.unique(a3))
            print(f"  X shape: {X.shape}, a3 values: {a3_unique}", flush=True)

            for a3_idx, a3_val in enumerate(a3_unique):
                mask = a3 == a3_val
                X_a, y_a = X[mask], y[mask]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_a,
                    y_a,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_a,
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                for clf_name, clf in make_classifiers(dset, n_jobs).items():
                    print(
                        f"  fit {dset}/{clf_name} a3={a3_val:.4f} "
                        f"(train={len(X_train)}, test={len(X_test)}, F={X_train.shape[1]})",
                        flush=True,
                    )
                    t0 = time.time()
                    clf.fit(X_train, y_train)
                    train_time = time.time() - t0

                    metrics, y_pred, y_score = evaluate_fit(
                        clf, X_train, y_train, X_test, y_test
                    )
                    print(
                        f"    test_acc={metrics['test_accuracy']:.4f}  "
                        f"auc={metrics['auc']:.4f}  "
                        f"time={train_time:.1f}s",
                        flush=True,
                    )

                    rows.append(
                        {
                            "dataset": dset,
                            "classifier": clf_name,
                            "a3": float(a3_val),
                            "a3_idx": int(a3_idx),
                            "n_train": len(X_train),
                            "n_test": len(X_test),
                            **metrics,
                            "train_time_sec": float(train_time),
                        }
                    )

                    grp = preds_f.create_group(f"{dset}/{clf_name}/a3_{a3_idx}")
                    grp.attrs["a3_value"] = float(a3_val)
                    grp.create_dataset("y_test", data=y_test.astype(np.uint8))
                    grp.create_dataset("y_pred", data=y_pred.astype(np.uint8))
                    grp.create_dataset("y_score", data=y_score.astype(np.float64))

    df = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "classifier",
            "a3",
            "a3_idx",
            "n_train",
            "n_test",
            "train_accuracy",
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
        default="/work/pi_mduarte_umass_edu/oraftery_umass_edu/results/classical_ml",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
