import argparse
import warnings
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from plot_style import WONG_PALETTE, save_fig, setup_style

ML_CLASSIFIERS = ("logreg", "svm", "rf", "gb")
THRESHOLD_STATS = ("power", "kurtosis", "papr", "autocorr", "regrowth_band_power")
DATASETS = ("bins", "feat", "iq", "spec")
DISPLAY_NAME = {
    "logreg": "LogReg",
    "svm": "SVM",
    "rf": "Random Forest",
    "gb": "Grad. Boost",
    "mlp": "MLP",
    "cnn": "1D CNN",
    "power": "Power",
    "kurtosis": "Kurtosis",
    "papr": "PAPR",
    "autocorr": "Autocorr",
    "regrowth_band_power": "Regrowth-band power",
}
DATASET_DISPLAY = {
    "bins": "averaged PSD (/bins, 100x avg)",
    "feat": "engineered features (/feat)",
    "iq": "raw I/Q (/iq, single realization)",
    "spec": "single-realization PSD (/spec = |FFT(iq)|^2)",
    "iq_flat": "raw I/Q flattened (MLP on /iq)",
}
COLORS = {
    "logreg": WONG_PALETTE[2],
    "svm": WONG_PALETTE[3],
    "rf": WONG_PALETTE[5],
    "gb": WONG_PALETTE[7],
    "mlp": "#000000",
    "cnn": "#000000",
    "power": WONG_PALETTE[1],
    "kurtosis": WONG_PALETTE[6],
    "papr": "#777777",
    "autocorr": "#333333",
    "regrowth_band_power": "#A50026",
}
LINESTYLES = {
    **{k: "-" for k in ("logreg", "svm", "rf", "gb", "mlp", "cnn")},
    **{k: "--" for k in THRESHOLD_STATS},
}
MARKERS = {
    "logreg": "o",
    "svm": "s",
    "rf": "^",
    "gb": "D",
    "mlp": "P",
    "cnn": "P",
    "power": "x",
    "kurtosis": "+",
    "papr": "*",
    "autocorr": "v",
    "regrowth_band_power": "X",
}


class Results:
    def __init__(self, results_dir: Path):
        self.dir = Path(results_dir)
        self.ml_metrics = self._try_csv("classical_ml/metrics.csv")
        self.nn_metrics = self._try_csv("nn/metrics.csv")
        self.th_metrics = self._try_csv("threshold_baselines/metrics.csv")
        self.ml_preds_path = self._maybe(self.dir / "classical_ml" / "predictions.h5")
        self.nn_preds_path = self._maybe(self.dir / "nn" / "predictions.h5")
        self.th_preds_path = self._maybe(
            self.dir / "threshold_baselines" / "predictions.h5"
        )

    def _try_csv(self, rel):
        path = self.dir / rel
        if path.exists():
            return pd.read_csv(path)
        warnings.warn(f"Missing: {path} -- related plots will be skipped.")
        return None

    @staticmethod
    def _maybe(p):
        return p if p.exists() else None


def plot_accuracy_vs_a3(dataset, results: Results, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if results.ml_metrics is not None:
        for clf in ML_CLASSIFIERS:
            sub = results.ml_metrics[
                (results.ml_metrics.dataset == dataset)
                & (results.ml_metrics.classifier == clf)
            ].sort_values("a3", key=lambda s: s.abs())
            if len(sub) == 0:
                continue
            ax.plot(
                sub.a3.abs(),
                sub.test_accuracy,
                label=DISPLAY_NAME[clf],
                color=COLORS[clf],
                linestyle=LINESTYLES[clf],
                marker=MARKERS[clf],
                markersize=6,
                linewidth=1.5,
            )
    nn_arch = "cnn" if dataset == "iq" else "mlp"
    if results.nn_metrics is not None:
        sub = results.nn_metrics[
            (results.nn_metrics.dataset == dataset)
            & (results.nn_metrics.architecture == nn_arch)
        ].sort_values("a3", key=lambda s: s.abs())
        if len(sub) > 0:
            ax.plot(
                sub.a3.abs(),
                sub.test_accuracy,
                label=DISPLAY_NAME[nn_arch],
                color=COLORS[nn_arch],
                linestyle=LINESTYLES[nn_arch],
                marker=MARKERS[nn_arch],
                markersize=7,
                linewidth=2.0,
            )
    if dataset == "iq" and results.nn_metrics is not None:
        sub = results.nn_metrics[
            (results.nn_metrics.dataset == "iq_flat")
            & (results.nn_metrics.architecture == "mlp")
        ].sort_values("a3", key=lambda s: s.abs())
        if len(sub) > 0:
            ax.plot(
                sub.a3.abs(),
                sub.test_accuracy,
                label="MLP (flat I/Q)",
                color="#555555",
                linestyle="-",
                marker="v",
                markersize=6,
                linewidth=1.5,
            )
    if dataset != "feat" and results.th_metrics is not None:
        for stat in THRESHOLD_STATS:
            sub = results.th_metrics[
                (results.th_metrics.dataset == dataset)
                & (results.th_metrics.statistic == stat)
            ].sort_values("a3", key=lambda s: s.abs())
            if len(sub) == 0:
                continue
            ax.plot(
                sub.a3.abs(),
                sub.best_accuracy,
                label=DISPLAY_NAME[stat],
                color=COLORS[stat],
                linestyle=LINESTYLES[stat],
                marker=MARKERS[stat],
                markersize=5,
                linewidth=1.2,
            )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="random guess")
    ax.set_xlabel(r"$|a_3|$  (nonlinearity strength)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f"Accuracy vs. nonlinearity strength: {DATASET_DISPLAY[dataset]}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
    save_fig(fig, f"accuracy_vs_a3_{dataset}", out_dir)


def plot_bar_comparison(results: Results, a3_idx: int, out_dir: Path):
    if all(
        m is None for m in (results.ml_metrics, results.nn_metrics, results.th_metrics)
    ):
        warnings.warn("No metrics at all; skipping bar comparison plot.")
        return
    a3_value = None
    bars_mf = {}
    bars_raw = {}
    if results.ml_metrics is not None:
        m = results.ml_metrics[results.ml_metrics.a3_idx == a3_idx]
        if len(m) == 0:
            warnings.warn(f"No ML rows at a3_idx={a3_idx}; skipping bar plot.")
            return
        a3_value = float(m.a3.iloc[0])
        for clf in ML_CLASSIFIERS:
            row_mf = m[(m.classifier == clf) & (m.dataset == "bins")]
            row_raw = m[(m.classifier == clf) & (m.dataset == "iq")]
            if len(row_mf):
                bars_mf[clf] = float(row_mf.test_accuracy.iloc[0])
            if len(row_raw):
                bars_raw[clf] = float(row_raw.test_accuracy.iloc[0])
    if results.nn_metrics is not None:
        n = results.nn_metrics[results.nn_metrics.a3_idx == a3_idx]
        row_mf = n[(n.architecture == "mlp") & (n.dataset == "bins")]
        row_raw = n[(n.architecture == "cnn") & (n.dataset == "iq")]
        if len(row_mf):
            bars_mf["mlp"] = float(row_mf.test_accuracy.iloc[0])
        if len(row_raw):
            bars_raw["cnn"] = float(row_raw.test_accuracy.iloc[0])
    if results.th_metrics is not None:
        t = results.th_metrics[results.th_metrics.a3_idx == a3_idx]
        for stat in THRESHOLD_STATS:
            row_mf = t[(t.statistic == stat) & (t.dataset == "bins")]
            row_raw = t[(t.statistic == stat) & (t.dataset == "iq")]
            if len(row_mf):
                bars_mf[stat] = float(row_mf.best_accuracy.iloc[0])
            if len(row_raw):
                bars_raw[stat] = float(row_raw.best_accuracy.iloc[0])
    algorithms = list(ML_CLASSIFIERS)
    nn_label_mf = "mlp" if "mlp" in bars_mf else None
    nn_label_raw = "cnn" if "cnn" in bars_raw else None
    if nn_label_mf or nn_label_raw:
        algorithms.append("nn")
    algorithms.extend(THRESHOLD_STATS)
    x = np.arange(len(algorithms))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mf_vals = []
    raw_vals = []
    labels_x = []
    for alg in algorithms:
        if alg == "nn":
            mf_v = bars_mf.get("mlp", np.nan)
            raw_v = bars_raw.get("cnn", np.nan)
            labels_x.append("NN\n(MLP/CNN)")
        else:
            mf_v = bars_mf.get(alg, np.nan)
            raw_v = bars_raw.get(alg, np.nan)
            labels_x.append(DISPLAY_NAME[alg])
        mf_vals.append(mf_v)
        raw_vals.append(raw_v)
    ax.bar(
        x - width / 2,
        mf_vals,
        width,
        label="Ensemble Averaging Willie (/bins)",
        color=WONG_PALETTE[5],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        raw_vals,
        width,
        label="Raw-IQ Willie (/iq)",
        color=WONG_PALETTE[6],
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=20, ha="right")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f"Per-method comparison at a3 = {a3_value}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    save_fig(fig, f"bar_comparison_a3_{a3_value}", out_dir)


def plot_training_curves(results: Results, out_dir: Path):
    if results.nn_preds_path is None:
        warnings.warn("No NN predictions.h5; skipping training curves.")
        return
    with h5py.File(results.nn_preds_path, "r") as f:
        for arch in f.keys():
            for dset in f[arch].keys():
                for a3_key in f[f"{arch}/{dset}"].keys():
                    grp = f[f"{arch}/{dset}/{a3_key}"]
                    a3_value = float(grp.attrs["a3_value"])
                    train_loss = grp["train_loss"][:]
                    val_loss = grp["val_loss"][:]
                    train_acc = grp["train_acc"][:]
                    val_acc = grp["val_acc"][:]
                    epochs = np.arange(1, len(train_loss) + 1)
                    fig, (ax_loss, ax_acc) = plt.subplots(
                        2, 1, figsize=(7, 5.2), sharex=True
                    )
                    ax_loss.plot(
                        epochs,
                        train_loss,
                        color=WONG_PALETTE[2],
                        linewidth=1.4,
                        label="train",
                    )
                    ax_loss.plot(
                        epochs,
                        val_loss,
                        color=WONG_PALETTE[6],
                        linestyle="--",
                        linewidth=1.4,
                        label="val",
                    )
                    ax_loss.set_ylabel("BCE loss")
                    ax_loss.legend(loc="upper right")
                    ax_loss.set_title(
                        f"{DISPLAY_NAME[arch]} on {DATASET_DISPLAY[dset]}, "
                        f"a3 = {a3_value}"
                    )
                    ax_acc.plot(
                        epochs,
                        train_acc,
                        color=WONG_PALETTE[2],
                        linewidth=1.4,
                        label="train",
                    )
                    ax_acc.plot(
                        epochs,
                        val_acc,
                        color=WONG_PALETTE[6],
                        linestyle="--",
                        linewidth=1.4,
                        label="val",
                    )
                    ax_acc.set_xlabel("epoch")
                    ax_acc.set_ylabel("accuracy")
                    ax_acc.legend(loc="lower right")
                    save_fig(
                        fig, f"training_curves_{arch}_{dset}_a3_{a3_value}", out_dir
                    )


def plot_confusion_matrix_from_counts(tn, fp, fn, tp, title, name, out_dir):
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums, 0.0)
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["H0", "H1"])
    ax.set_yticklabels(["H0", "H1"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1%})",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    save_fig(fig, name, out_dir)


def plot_all_confusion_matrices(results: Results, out_dir: Path):
    out = out_dir / "confusion_matrices"
    if results.ml_metrics is not None:
        for _, r in results.ml_metrics.iterrows():
            plot_confusion_matrix_from_counts(
                r.tn,
                r.fp,
                r.fn,
                r.tp,
                f"{DISPLAY_NAME[r.classifier]}, /{r.dataset}, a3={r.a3}",
                f"cm_{r.classifier}_{r.dataset}_a3_{r.a3}",
                out,
            )
    if results.nn_metrics is not None:
        for _, r in results.nn_metrics.iterrows():
            plot_confusion_matrix_from_counts(
                r.tn,
                r.fp,
                r.fn,
                r.tp,
                f"{DISPLAY_NAME[r.architecture]}, /{r.dataset}, a3={r.a3}",
                f"cm_{r.architecture}_{r.dataset}_a3_{r.a3}",
                out,
            )
    if results.th_metrics is not None:
        for _, r in results.th_metrics.iterrows():
            plot_confusion_matrix_from_counts(
                r.tn,
                r.fp,
                r.fn,
                r.tp,
                f"{DISPLAY_NAME[r.statistic]}, /{r.dataset}, a3={r.a3}",
                f"cm_{r.statistic}_{r.dataset}_a3_{r.a3}",
                out,
            )


def _draw_roc(ax, y_test, y_score, label, color, linestyle, linewidth=1.4):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_val = auc(fpr, tpr)
    ax.plot(
        fpr,
        tpr,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=f"{label} (AUC={auc_val:.3f})",
    )


def plot_roc_curves(results: Results, out_dir: Path):
    if all(
        p is None
        for p in (results.ml_preds_path, results.nn_preds_path, results.th_preds_path)
    ):
        warnings.warn("No predictions HDF5s available; skipping ROC curves.")
        return
    a3_map = {}
    for path in (results.ml_preds_path, results.nn_preds_path, results.th_preds_path):
        if path is None:
            continue
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if key in DATASETS:
                    dataset = key
                    for sub in f[key].keys():
                        for a3_key, grp in f[f"{key}/{sub}"].items():
                            a3_idx = int(a3_key.split("_")[1])
                            a3_map.setdefault(dataset, {})[a3_idx] = float(
                                grp.attrs["a3_value"]
                            )
                else:
                    for dataset in f[key].keys():
                        for a3_key, grp in f[f"{key}/{dataset}"].items():
                            a3_idx = int(a3_key.split("_")[1])
                            a3_map.setdefault(dataset, {})[a3_idx] = float(
                                grp.attrs["a3_value"]
                            )
    for dataset, idx_to_value in a3_map.items():
        for a3_idx, a3_value in sorted(idx_to_value.items()):
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax.set_aspect("equal", adjustable="box")
            ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8)
            if results.ml_preds_path is not None:
                with h5py.File(results.ml_preds_path, "r") as f:
                    for clf in ML_CLASSIFIERS:
                        path = f"{dataset}/{clf}/a3_{a3_idx}"
                        if path in f:
                            grp = f[path]
                            _draw_roc(
                                ax,
                                grp["y_test"][:],
                                grp["y_score"][:],
                                DISPLAY_NAME[clf],
                                COLORS[clf],
                                LINESTYLES[clf],
                            )
            nn_arch = "cnn" if dataset == "iq" else "mlp"
            if results.nn_preds_path is not None:
                with h5py.File(results.nn_preds_path, "r") as f:
                    path = f"{nn_arch}/{dataset}/a3_{a3_idx}"
                    if path in f:
                        grp = f[path]
                        _draw_roc(
                            ax,
                            grp["y_test"][:],
                            grp["y_score"][:],
                            DISPLAY_NAME[nn_arch],
                            COLORS[nn_arch],
                            LINESTYLES[nn_arch],
                            linewidth=1.8,
                        )
            if dataset != "feat" and results.th_preds_path is not None:
                with h5py.File(results.th_preds_path, "r") as f:
                    for stat in THRESHOLD_STATS:
                        path = f"{dataset}/{stat}/a3_{a3_idx}"
                        if path in f:
                            grp = f[path]
                            _draw_roc(
                                ax,
                                grp["y_test"][:],
                                grp["y_score"][:],
                                DISPLAY_NAME[stat],
                                COLORS[stat],
                                LINESTYLES[stat],
                            )
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"ROC: {DATASET_DISPLAY[dataset]}, a3 = {a3_value}")
            ax.legend(
                loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=True
            )
            save_fig(fig, f"roc_{dataset}_a3_{a3_value}", out_dir)


def write_table(df, name, out_dir, float_fmt="%.4f"):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}.csv"
    tex = out_dir / f"{name}.tex"
    df.to_csv(csv, index=False, float_format=float_fmt)
    try:
        df.to_latex(tex, index=False, float_format=float_fmt, escape=False)
    except Exception as e:
        warnings.warn(f"Could not write LaTeX for {name}: {e}")
    print(f"  wrote {csv}")
    print(f"  wrote {tex}")


def make_tables(results: Results, out_dir: Path):
    frames = []
    if results.ml_metrics is not None:
        frames.append(
            results.ml_metrics.assign(
                method=results.ml_metrics.classifier.map(DISPLAY_NAME),
                family="ML",
            )[["method", "family", "dataset", "a3", "test_accuracy"]]
        )
    if results.nn_metrics is not None:
        frames.append(
            results.nn_metrics.assign(
                method=results.nn_metrics.architecture.map(DISPLAY_NAME),
                family="NN",
            )[["method", "family", "dataset", "a3", "test_accuracy"]]
        )
    if results.th_metrics is not None:
        frames.append(
            results.th_metrics.assign(
                method=results.th_metrics.statistic.map(DISPLAY_NAME),
                family="Threshold",
                test_accuracy=results.th_metrics.best_accuracy,
            )[["method", "family", "dataset", "a3", "test_accuracy"]]
        )
    if frames:
        master = pd.concat(frames, ignore_index=True)
        pivot = master.pivot_table(
            index=["family", "method", "dataset"],
            columns="a3",
            values="test_accuracy",
        ).reset_index()
        write_table(pivot, "master_accuracy", out_dir)
    if results.ml_metrics is not None:
        for dset in DATASETS:
            sub = results.ml_metrics[results.ml_metrics.dataset == dset]
            if len(sub) == 0:
                continue
            agg = sub.groupby("classifier", as_index=False).agg(
                mean_train_accuracy=("train_accuracy", "mean"),
                mean_test_accuracy=("test_accuracy", "mean"),
                mean_auc=("auc", "mean"),
                mean_train_time_sec=("train_time_sec", "mean"),
            )
            agg["classifier"] = agg["classifier"].map(DISPLAY_NAME)
            write_table(agg, f"per_dataset_summary_{dset}", out_dir)
        if results.nn_metrics is not None:
            for dset in DATASETS:
                sub = results.nn_metrics[results.nn_metrics.dataset == dset]
                if len(sub) == 0:
                    continue
                row = sub.groupby("architecture", as_index=False).agg(
                    mean_best_val_acc=("best_val_acc", "mean"),
                    mean_test_accuracy=("test_accuracy", "mean"),
                    mean_auc=("auc", "mean"),
                    mean_train_time_sec=("train_time_sec", "mean"),
                )
                row["architecture"] = row["architecture"].map(DISPLAY_NAME)
                write_table(row, f"per_dataset_summary_nn_{dset}", out_dir)
    if results.th_metrics is not None:
        pivot = results.th_metrics.pivot_table(
            index=["dataset", "statistic"],
            columns="a3",
            values="best_accuracy",
        ).reset_index()
        pivot["statistic"] = pivot["statistic"].map(DISPLAY_NAME)
        write_table(pivot, "threshold_baseline_summary", out_dir)


def make_all(results: Results, fig_dir: Path, table_dir: Path, a3_idx_for_bar: int):
    print("\n=== accuracy_vs_a3 ===")
    for dset in DATASETS:
        plot_accuracy_vs_a3(dset, results, fig_dir)
    print("\n=== bar_comparison ===")
    plot_bar_comparison(results, a3_idx_for_bar, fig_dir)
    print("\n=== training_curves ===")
    plot_training_curves(results, fig_dir)
    print("\n=== confusion_matrices ===")
    plot_all_confusion_matrices(results, fig_dir)
    print("\n=== roc_curves ===")
    plot_roc_curves(results, fig_dir)
    print("\n=== tables ===")
    make_tables(results, table_dir)


CATEGORIES = {
    "accuracy_vs_a3": lambda r, f, t, a: [
        plot_accuracy_vs_a3(d, r, f) for d in DATASETS
    ],
    "bar_comparison": lambda r, f, t, a: plot_bar_comparison(r, a, f),
    "training_curves": lambda r, f, t, a: plot_training_curves(r, f),
    "confusion_matrices": lambda r, f, t, a: plot_all_confusion_matrices(r, f),
    "roc_curves": lambda r, f, t, a: plot_roc_curves(r, f),
    "tables": lambda r, f, t, a: make_tables(r, t),
    "all": lambda r, f, t, a: make_all(r, f, t, a),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing classical_ml/, nn/, threshold_baselines/ subdirs.",
    )
    p.add_argument("--fig-dir", default="figures/results", type=Path)
    p.add_argument("--table-dir", default="tables", type=Path)
    p.add_argument(
        "--a3-idx-for-bar",
        type=int,
        default=2,
        help="Which a3 index to use for the bar-comparison plot (default: 2 = middle of 5-value sweep)",
    )
    p.add_argument("--only", default="all", choices=list(CATEGORIES))
    args = p.parse_args()
    setup_style()
    results = Results(Path(args.results_dir))
    CATEGORIES[args.only](results, args.fig_dir, args.table_dir, args.a3_idx_for_bar)
    print("\nDone.")


if __name__ == "__main__":
    main()
