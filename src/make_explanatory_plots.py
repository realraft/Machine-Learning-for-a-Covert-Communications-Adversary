import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from plot_style import (
    LINESTYLES,
    WONG_PALETTE,
    save_fig,
    setup_style,
    viridis_n,
)

N_SYM = 100
N_FFT = 100
BETA = 0.25
SPAN = 5
OSR = 16
A1 = 1.0
A3_VALUES = (-0.08, -0.16, -0.24, -0.32, -0.40)
NOISE_RATIO = 0.5
DEFAULT_OUT_DIR = Path("figures/explanatory")


def srrc_pulse(beta=BETA, span=SPAN, osr=OSR):
    n_samples = span * osr
    t = np.arange(-n_samples // 2, n_samples // 2 + 1) / osr
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 + beta * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / math.sqrt(2)) * (
                (1 + 2 / np.pi) * math.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * math.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))


def gen_gaussian_symbols(n_sym, rng):
    ak = (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)) / np.sqrt(2)
    ak[:5] = 0
    ak[-5:] = 0
    return ak


def upsample(x, factor):
    out = np.zeros(len(x) * factor, dtype=x.dtype)
    out[::factor] = x
    return out


def make_qpsk_signal(rng):
    h = srrc_pulse()
    ak = gen_gaussian_symbols(N_SYM, rng)
    ak_up = upsample(ak, OSR)
    return np.convolve(ak_up, h, mode="same")


def simulate_one(a3, rng, add_noise=True, normalize=True):
    h = srrc_pulse()
    alice = make_qpsk_signal(rng)
    alice_NL = A1 * alice + a3 * (alice**3)
    jammer = make_qpsk_signal(rng)
    sig_H0 = jammer
    sig_H1 = alice_NL + jammer
    if normalize:
        sig_H0 = sig_H0 / np.sqrt(np.mean(np.abs(sig_H0) ** 2) + 1e-12)
        sig_H1 = sig_H1 / np.sqrt(np.mean(np.abs(sig_H1) ** 2) + 1e-12)
    if add_noise:
        n = len(sig_H0)
        s = math.sqrt(NOISE_RATIO / 2)
        noise_H0 = s * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        noise_H1 = s * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        sig_H0 = sig_H0 + noise_H0
        sig_H1 = sig_H1 + noise_H1
    return sig_H0, sig_H1, h


def extract_features(sig):
    sig = np.asarray(sig).flatten()
    env = np.abs(sig)
    crest_factor = env.max() / (env.mean() + 1e-10)
    env_norm = (env - env.mean()) / (env.std() + 1e-10)
    envelope_skewness = float(np.mean(env_norm**3))
    envelope_kurtosis = float(np.mean(env_norm**4))
    peak_power = 10 * float(np.log10(np.max(np.abs(sig) ** 2) + 1e-10))
    m2 = float(np.mean(np.abs(sig) ** 2))
    m4 = float(np.mean(np.abs(sig) ** 4))
    normalized_m4 = m4 / (m2**2 + 1e-10)
    P = np.abs(np.fft.fft(sig)) ** 2 / len(sig)
    P_norm = P / (P.mean() + 1e-10)
    spectral_flatness = float(np.exp(np.mean(np.log(P_norm + 1e-10))))
    P_pdf = P / (P.sum() + 1e-10)
    spectral_entropy = float(-np.sum(P_pdf * np.log2(P_pdf + 1e-10)))
    return np.array(
        [
            crest_factor,
            envelope_skewness,
            envelope_kurtosis,
            peak_power,
            normalized_m4,
            spectral_flatness,
            spectral_entropy,
        ]
    )


def averaged_psd(a3, n_real, rng, hypothesis="H1", add_noise=False, nperseg=128):
    if hypothesis not in ("H0", "H1"):
        raise ValueError(f"hypothesis must be 'H0' or 'H1', got {hypothesis}")
    psd_acc = None
    f_axis = None
    for _ in range(n_real):
        sig_H0, sig_H1, _ = simulate_one(a3, rng, add_noise=add_noise)
        sig = sig_H1 if hypothesis == "H1" else sig_H0
        f_w, psd_w = welch(
            sig,
            fs=OSR,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=False,
            scaling="density",
            detrend=False,
        )
        f_w = np.fft.fftshift(f_w)
        psd_w = np.fft.fftshift(psd_w)
        if psd_acc is None:
            psd_acc = psd_w
            f_axis = f_w
        else:
            psd_acc += psd_w
    psd_acc /= n_real
    return f_axis, psd_acc


def plot_srrc_pulse(out_dir=DEFAULT_OUT_DIR):
    h = srrc_pulse()
    n = len(h)
    t = (np.arange(n) - n // 2) / OSR
    fig, ax = plt.subplots()
    ax.plot(t, h, color=WONG_PALETTE[5], linewidth=1.6)
    for k in range(-SPAN // 2, SPAN // 2 + 1):
        ax.axvline(k, color="gray", linestyle=":", alpha=0.5, linewidth=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("time / symbol period")
    ax.set_ylabel(r"$p(t)$")
    ax.set_title(rf"SRRC pulse  ($\beta$ = {BETA}, span = {SPAN} symbols)")
    save_fig(fig, "srrc_pulse", out_dir)


def plot_pulse_train(out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    n_show = 8
    h = srrc_pulse()
    syms = (2 * rng.integers(0, 2, n_show) - 1).astype(float)
    syms_up = np.zeros(n_show * OSR)
    syms_up[::OSR] = syms
    composite = np.convolve(syms_up, h, mode="same")
    t = (np.arange(len(composite)) - len(h) // 2) / OSR
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for k in range(n_show):
        single = np.zeros_like(syms_up)
        single[k * OSR] = syms[k]
        single_filtered = np.convolve(single, h, mode="same")
        ax.plot(t, single_filtered, color=WONG_PALETTE[2], alpha=0.45, linewidth=1.0)
    ax.plot(t, composite, color=WONG_PALETTE[6], linewidth=1.8, label="composite (sum)")
    ax.scatter(
        np.arange(n_show),
        syms,
        marker="o",
        color="black",
        s=22,
        zorder=5,
        label="symbol values",
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("time / symbol period")
    ax.set_ylabel("amplitude")
    ax.set_title(
        "Pulse train: individual SRRC pulses (light) summing to the transmitted waveform"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    save_fig(fig, "pulse_train", out_dir)


def plot_linear_vs_nonlinear_time(a3=A3_VALUES[3], out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    sig_H0, sig_H1, _ = simulate_one(a3, rng, add_noise=False)
    start = N_SYM * OSR // 4
    end = start + 5 * OSR
    t = np.arange(start, end) / OSR
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(
        t,
        np.real(sig_H0[start:end]),
        color=WONG_PALETTE[5],
        label="H0 (Alice absent)",
        linewidth=1.5,
    )
    axes[0].plot(
        t,
        np.real(sig_H1[start:end]),
        color=WONG_PALETTE[6],
        linestyle="--",
        label="H1 (Alice present)",
        linewidth=1.5,
    )
    axes[0].set_ylabel(r"Re$\{r(t)\}$")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"H0 vs. H1 received baseband signal  (a3 = {a3})")
    axes[1].plot(
        t,
        np.imag(sig_H0[start:end]),
        color=WONG_PALETTE[5],
        label="H0 (Alice absent)",
        linewidth=1.5,
    )
    axes[1].plot(
        t,
        np.imag(sig_H1[start:end]),
        color=WONG_PALETTE[6],
        linestyle="--",
        label="H1 (Alice present)",
        linewidth=1.5,
    )
    axes[1].set_ylabel(r"Im$\{r(t)\}$")
    axes[1].set_xlabel("time / symbol period")
    axes[1].legend(loc="upper right")
    save_fig(fig, f"h0_vs_h1_time_a3_{a3}", out_dir)


def plot_linear_vs_nonlinear_psd(a3=A3_VALUES[3], n_real=200, out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    f, psd_H0 = averaged_psd(a3, n_real, rng, hypothesis="H0")
    _, psd_H1 = averaged_psd(a3, n_real, rng, hypothesis="H1")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        f,
        10 * np.log10(psd_H0 + 1e-20),
        color=WONG_PALETTE[5],
        label="H0 (Alice absent)",
        linewidth=1.5,
    )
    ax.plot(
        f,
        10 * np.log10(psd_H1 + 1e-20),
        color=WONG_PALETTE[6],
        linestyle="--",
        label="H1 (Alice present)",
        linewidth=1.5,
    )
    band_edge = (1 + BETA) / 2
    ax.axvspan(-band_edge, band_edge, alpha=0.07, color="gray")
    ax.text(
        0,
        ax.get_ylim()[1],
        " main band  ",
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )
    ax.set_xlabel("frequency / symbol rate")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(
        f"H0 vs. H1 PSD after unit-power normalization  (a3 = {a3}, "
        f"Welch avg of {n_real} realizations)"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    save_fig(fig, f"h0_vs_h1_psd_a3_{a3}", out_dir)


PSD_UNNORM_A3 = -1.0


def plot_psd_unnormalized(a3=PSD_UNNORM_A3, n_real=200, out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    h = srrc_pulse()
    psd_H0_acc = None
    psd_H1_acc = None
    f_axis = None
    for _ in range(n_real):
        alice = make_qpsk_signal(rng)
        alice_NL = A1 * alice + a3 * (alice**3)
        jammer = make_qpsk_signal(rng)
        sig_H0 = jammer
        sig_H1 = alice_NL + jammer
        for sig, slot in ((sig_H0, "H0"), (sig_H1, "H1")):
            f_w, psd_w = welch(
                sig,
                fs=OSR,
                nperseg=128,
                noverlap=64,
                return_onesided=False,
                scaling="density",
                detrend=False,
            )
            psd_w = np.fft.fftshift(psd_w)
            if slot == "H0":
                psd_H0_acc = psd_w if psd_H0_acc is None else psd_H0_acc + psd_w
            else:
                psd_H1_acc = psd_w if psd_H1_acc is None else psd_H1_acc + psd_w
            if f_axis is None:
                f_axis = np.fft.fftshift(f_w)
    psd_H0_acc /= n_real
    psd_H1_acc /= n_real
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        f_axis,
        10 * np.log10(psd_H0_acc + 1e-20),
        color=WONG_PALETTE[5],
        label="H0 (jammer alone)",
        linewidth=1.5,
    )
    ax.plot(
        f_axis,
        10 * np.log10(psd_H1_acc + 1e-20),
        color=WONG_PALETTE[6],
        linestyle="--",
        label="H1 (Alice + jammer, regrowth visible)",
        linewidth=1.5,
    )
    band_edge = (1 + BETA) / 2
    ax.axvspan(-band_edge, band_edge, alpha=0.07, color="gray")
    ax.text(
        0,
        ax.get_ylim()[1],
        " main band  ",
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )
    ax.set_xlabel("frequency / symbol rate")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(
        f"Spectral regrowth before power equalization  "
        f"(a3 = {a3}, Welch avg of {n_real} realizations)"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    save_fig(fig, f"psd_unnormalized_a3_{a3}", out_dir)


def plot_psd_vs_a3(n_real=100, out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    f, psd_H0 = averaged_psd(0.0, n_real, rng, hypothesis="H0")
    a3_values_local = [-0.08, -0.16, -0.24, -0.32, -0.45, -0.60, -0.80]
    colors = viridis_n(len(a3_values_local))
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(
        f,
        10 * np.log10(psd_H0 + 1e-20),
        color="black",
        linewidth=2.0,
        label="H0 (Alice absent)",
        linestyle="-",
    )
    for i, a3 in enumerate(a3_values_local):
        _, psd = averaged_psd(a3, n_real, rng, hypothesis="H1")
        ax.plot(
            f,
            10 * np.log10(psd + 1e-20),
            color=colors[i],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            linewidth=1.5,
            label=f"H1, a3 = {a3}",
        )
    ax.set_xlabel("frequency / symbol rate")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(
        f"H1 PSD across Alice's a3 sweep  (Welch avg of {n_real} realizations)"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
    save_fig(fig, "psd_vs_a3", out_dir)


def plot_matched_filter(out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    _, sig_H1, h = simulate_one(A3_VALUES[3], rng, add_noise=True)
    x = sig_H1
    y = np.convolve(x, h, mode="same")
    start = N_SYM * OSR // 4
    end = start + 8 * OSR
    t = np.arange(start, end) / OSR
    th = (np.arange(len(h)) - len(h) // 2) / OSR
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5))
    axes[0].plot(t, np.real(x[start:end]), color=WONG_PALETTE[5], linewidth=1.2)
    axes[0].set_ylabel(r"Re$\{r(t)\}$")
    axes[0].set_title("(a) Noisy received signal")
    axes[1].plot(th, h, color=WONG_PALETTE[6])
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel(r"$h(t)$")
    axes[1].set_title("(b) Matched filter impulse response (SRRC)")
    axes[2].plot(t, np.real(y[start:end]), color=WONG_PALETTE[2], linewidth=1.5)
    for sym_idx in range(start // OSR + 1, end // OSR):
        axes[2].axvline(sym_idx, color="gray", linestyle=":", alpha=0.4, linewidth=0.6)
    axes[2].set_ylabel(r"Re$\{y(t)\}$")
    axes[2].set_xlabel("time / symbol period")
    axes[2].set_title(
        "(c) Matched filter output (dotted lines: symbol-decision instants)"
    )
    fig.tight_layout()
    save_fig(fig, "matched_filter_input_output", out_dir)


def plot_fft_averaging(a3=A3_VALUES[3], out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    f_single, psd_single = averaged_psd(
        a3, n_real=1, rng=rng, hypothesis="H1", add_noise=True
    )
    f_avg, psd_avg = averaged_psd(
        a3, n_real=N_FFT, rng=rng, hypothesis="H1", add_noise=True
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(
        f_single,
        10 * np.log10(psd_single + 1e-20),
        color=WONG_PALETTE[2],
        alpha=0.8,
        linewidth=0.9,
        label="single periodogram (n_real = 1)",
    )
    ax.plot(
        f_avg,
        10 * np.log10(psd_avg + 1e-20),
        color=WONG_PALETTE[6],
        linewidth=1.8,
        label=f"averaged over n_real = {N_FFT}",
    )
    ax.set_xlabel("frequency / symbol rate")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(f"Effect of ensemble averaging on H1 PSD estimate  (a3 = {a3})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    save_fig(fig, f"fft_averaging_a3_{a3}", out_dir)


def plot_features_distribution(a3=A3_VALUES[3], n_runs=1000, out_dir=DEFAULT_OUT_DIR):
    rng = np.random.default_rng(42)
    feature_names = [
        "crest_factor",
        "envelope_skewness",
        "envelope_kurtosis",
        "peak_power",
        "normalized_m4",
        "spectral_flatness",
        "spectral_entropy",
    ]
    feats_H0 = np.zeros((n_runs, 7))
    feats_H1 = np.zeros((n_runs, 7))
    for r in range(n_runs):
        sig_H0, sig_H1, _ = simulate_one(a3, rng, add_noise=True)
        feats_H0[r] = extract_features(sig_H0)
        feats_H1[r] = extract_features(sig_H1)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    axes_flat = axes.flatten()
    for i, name in enumerate(feature_names):
        ax = axes_flat[i]
        lo = min(feats_H0[:, i].min(), feats_H1[:, i].min())
        hi = max(feats_H0[:, i].max(), feats_H1[:, i].max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(
            feats_H0[:, i],
            bins=bins,
            color=WONG_PALETTE[5],
            alpha=0.55,
            label="H0 (Alice absent)",
            edgecolor="none",
        )
        ax.hist(
            feats_H1[:, i],
            bins=bins,
            color=WONG_PALETTE[6],
            alpha=0.55,
            label="H1 (Alice present)",
            edgecolor="none",
        )
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes_flat[7].set_visible(False)
    fig.suptitle(
        f"Engineered feature distributions  (a3 = {a3}, n = {n_runs} per class)"
    )
    fig.tight_layout()
    save_fig(fig, f"features_distribution_a3_{a3}", out_dir)


PLOT_FUNCTIONS = {
    "srrc_pulse": plot_srrc_pulse,
    "pulse_train": plot_pulse_train,
    "h0_vs_h1_time": plot_linear_vs_nonlinear_time,
    "h0_vs_h1_psd": plot_linear_vs_nonlinear_psd,
    "psd_unnormalized": plot_psd_unnormalized,
    "psd_vs_a3": plot_psd_vs_a3,
    "matched_filter": plot_matched_filter,
    "fft_averaging": plot_fft_averaging,
    "features_distribution": plot_features_distribution,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        default=None,
        choices=list(PLOT_FUNCTIONS),
        help="Generate a single plot instead of all of them.",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        type=Path,
        help="Output directory for figures (default: figures/explanatory)",
    )
    args = p.parse_args()
    setup_style()
    if args.only:
        print(f"Generating {args.only} ...")
        PLOT_FUNCTIONS[args.only](out_dir=args.out_dir)
    else:
        for name, fn in PLOT_FUNCTIONS.items():
            print(f"Generating {name} ...")
            fn(out_dir=args.out_dir)
    print(f"\nDone. Figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
