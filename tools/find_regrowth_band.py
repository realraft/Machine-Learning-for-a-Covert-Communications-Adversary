# this script intends to find the best bins in training data
# to showcase spectral regrowth (most separable)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bins_data.csv")
N_BINS = 800 # number of frequency bin columns
LABEL_COL = "nonlinear" # name of the class label column
TOP_N = 20 # number of top separable bins to highlight
CLUSTER_GAP = 10 # max gap (in bins) to consider two bins part of the same cluster
STD_ALPHA = 0.25 # transparency of ±1 std shaded bands

# load data and split by class
data = pd.read_csv(DATA_PATH)
feature_cols = data.columns[:N_BINS]
X = data[feature_cols].values
y = data[LABEL_COL].values

linear = y == 0
nonlinear = y == 1

linear_X = X[linear]
nonlinear_X = X[nonlinear]

# compute mean and std per class
mean_linear = linear_X.mean(axis=0)
mean_nonlinear = nonlinear_X.mean(axis=0)
std_linear = linear_X.std(axis=0)
std_nonlinear = nonlinear_X.std(axis=0)

# compute separability
sep = np.abs(mean_nonlinear - mean_linear)
top_indices_sorted = np.argsort(sep)[::-1] # sort separable bins
top20 = top_indices_sorted[:TOP_N]

print(f"Top {TOP_N} most separable bins: {top20.tolist()}")

# plots
bin_axis = np.arange(N_BINS)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# plot 1: mean power spectrum with ±1 std shaded regions
ax1.plot(bin_axis, mean_nonlinear, color="tomato", lw=1.5, label="Nonlinear")
ax1.fill_between(
    bin_axis,
    mean_nonlinear - std_nonlinear,
    mean_nonlinear + std_nonlinear,
    color="tomato", alpha=STD_ALPHA
)
ax1.plot(bin_axis, mean_linear, color="steelblue", linestyle='--', lw=1.5, label="Linear")
ax1.fill_between(
    bin_axis,
    mean_linear - std_linear,
    mean_linear + std_linear,
    color="steelblue", alpha=STD_ALPHA
)
ax1.set_xlabel("Bin Index")
ax1.set_ylabel("Mean Power (log scale)")
ax1.set_title("Mean Power Spectrum by Class — Full Band (±1σ shaded)")
ax1.set_yscale("log")
ax1.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

# plot 2: per-bin separability showing absolute difference between class means
ax2.plot(bin_axis, sep, color="darkorchid", lw=1.2, label="Separability")
ax2.set_xlabel("Bin Index")
ax2.set_ylabel("Mean Difference (log scale)")
ax2.set_title("Per-Bin Class Separability")
ax2.set_yscale("log")
ax2.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.show()
