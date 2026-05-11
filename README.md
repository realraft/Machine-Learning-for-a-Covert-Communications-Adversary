# Machine Learning for a Covert Communications Adversary

UMass Amherst ECE Departmental Honors Thesis, 2025–2026.
**Student:** Owen Raftery
**Advisor:** Prof. Dennis Goeckel

Alice transmits through a cubic-nonlinearity power amplifier (coefficient `a3`); Willie tries to detect her against a jammer + AWGN. Classifiers are trained at each `a3` and compared against threshold-test baselines (power, kurtosis, PAPR, autocorr).

**Willie variants:** PSD averaging, raw-I/Q, and feature engineering
**Labels:** `0` = linear, `1` = nonlinear.

## Layout
- `src/cluster_simulation.m` - MATLAB sim, generates all three datasets
- `src/train_classical_ml.py`, `src/train_nn.py` - classifiers
- `src/threshold_baselines.py` - single-statistic detectors
- `src/make_results_plots.py`, `src/make_explanatory_plots.py` - figures
- `scripts/cluster_*.sh` - Slurm jobs for Unity
