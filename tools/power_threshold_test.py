# this script sweeps possible power thresholds from the maximum and minimum
# power in the specified regrowth band to find the best accuracy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_PATH = os.path.join("data", "bins_data.csv") 
N_BINS = 800 # number of frequency bin columns
LABEL_COL = "nonlinear" # name of the class label column
REGROWTH_START = 50 # positional bin index (inclusive)
REGROWTH_END = 150 # positional bin index (inclusive)
N_THRESHOLDS = 1000 # number of evenly spaced sweep values

data = pd.read_csv(DATA_PATH)

features = data.iloc[:, :N_BINS].values
labels = data[LABEL_COL].values.astype(int)

# Compute per-signal regrowth band power
band = features[:, REGROWTH_START : REGROWTH_END + 1]
regrowth_power = np.sum(band, axis=1)

print("Mean regrowth power:")
print(f"Linear: {regrowth_power[labels == 0].mean():.6e}")
print(f"Nonlinear: {regrowth_power[labels == 1].mean():.6e}")

# threshold sweep
thresholds = np.linspace(regrowth_power.min(), regrowth_power.max(), N_THRESHOLDS) # array of thresholds from min to max regrowth power

# initialize arrays
accuracies = np.zeros(N_THRESHOLDS)
tprs = np.zeros(N_THRESHOLDS)
fprs = np.zeros(N_THRESHOLDS)

nonlinear = labels == 1
linear = labels == 0
n_nonlinear = nonlinear.sum()
n_linear = linear.sum()

# test every threshold
for i, T in enumerate(thresholds):
    preds = (regrowth_power < T).astype(int) # make predictions

    # record prediction accuracy through confidence matrix
    tp = int(( preds[nonlinear] == 1).sum())
    fn = int(( preds[nonlinear] == 0).sum())
    fp = int(( preds[linear] == 1).sum())
    tn = int(( preds[linear] == 0).sum())

    # store values for this run
    accuracies[i] = np.mean(preds == labels)
    tprs[i]       = tp / n_nonlinear if n_nonlinear > 0 else 0.0
    fprs[i]       = fp / n_linear if n_linear > 0 else 0.0

# locate best threshold and its accuracy
best_idx  = int(np.argmax(accuracies))
best_T    = thresholds[best_idx]
best_acc  = accuracies[best_idx]

print(f"\nBest threshold: {best_T:.6e}")

# final evaluation at T* (best threshold)
final_preds = (regrowth_power > best_T).astype(int)
final_acc = np.mean(final_preds == labels)

precision  = precision_score(labels, final_preds, zero_division=0)
recall_val = recall_score(labels, final_preds, zero_division=0)
f1         = f1_score(labels, final_preds, zero_division=0)

print(f"Accuracy: {final_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1 Score: {f1:.4f}")

# plots
fig, axes = plt.subplots(3, 1, figsize=(8, 14))

# power distribution histogram
ax1 = axes[0]
bins_hist = 40
ax1.hist(regrowth_power[labels == 0], bins=bins_hist, alpha=0.55,
         label="Linear", color="steelblue")
ax1.hist(regrowth_power[labels == 1], bins=bins_hist, alpha=0.55,
         label="Nonlinear", color="darkorange")
ax1.axvline(best_T, color="red", linestyle="--", linewidth=1.8, label=f"T* = {best_T:.2e}")
ax1.set_xlabel("Regrowth Band Power")
ax1.set_ylabel("Count")
ax1.set_title("Regrowth Power Distribution with Decision Threshold")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# accuracy vs threshold
ax2 = axes[1]
ax2.plot(thresholds, accuracies, color="steelblue", linewidth=1.5)
ax2.plot(best_T, best_acc, "ro", markersize=8, label=f"T* = {best_T:.2e}\nAcc = {best_acc:.4f}")
ax2.set_xlabel("Threshold T")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Across Threshold Sweep")
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# ROC curve
sort_idx = np.argsort(fprs)
fprs_s = fprs[sort_idx]
tprs_s = tprs[sort_idx]
auc_val = float(np.trapezoid(tprs_s, fprs_s))

op_fpr = fprs[best_idx]
op_tpr = tprs[best_idx]

ax3 = axes[2]
ax3.plot(fprs_s, tprs_s, color="steelblue", linewidth=1.8,
         label=f"ROC (AUC = {auc_val:.4f})")
ax3.plot(op_fpr, op_tpr, "ro", markersize=9,
         label=f"T* operating point\n(FPR={op_fpr:.3f}, TPR={op_tpr:.3f})")
ax3.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.2,
         label="Random classifier")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title(f"ROC Curve  (AUC = {auc_val:.4f})")
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(h_pad=4.0)
plt.show()
