from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WONG_PALETTE = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]
LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1, 1, 1)),
]


def setup_style():
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


def viridis_n(n):
    return [plt.cm.viridis(t) for t in np.linspace(0.15, 0.85, n)]
