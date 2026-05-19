"""
Generate confusion matrix V10 untuk keperluan skripsi.
Data diambil langsung dari hasil evaluasi di rangkuman_training.md.

Output (disimpan di phase3_training/outputs/):
  - cm_v10_default.png    (threshold 0.5)
  - cm_v10_calibrated.png (threshold 0.170)
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUTPUT_DIR = pathlib.Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Engaged", "NotEngaged"]

# Data dari rangkuman_training.md — V10 default (thr=0.5), n=4238
CM_DEFAULT = np.array([
    [1767,  861],   # True Engaged   → pred Engaged, pred NotEngaged
    [ 215, 1395],   # True NotEngaged → pred Engaged, pred NotEngaged
])

# Data dari rangkuman_training.md — V10 calibrated (thr=0.170), n=3390
CM_CALIBRATED = np.array([
    [1862,  240],
    [ 363,  925],
])


def plot_cm(cm: np.ndarray, title: str, out_path: pathlib.Path) -> None:
    """Plot confusion matrix ternormalisasi (row-wise = recall per kelas)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": ""},
        ax=ax,
    )

    # Tulis nilai di tiap sel
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm_norm[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(
                j + 0.5, i + 0.5,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=14, color=color, fontweight="bold",
            )

    ax.set_xlabel("True", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    plot_cm(
        CM_DEFAULT,
        title="Confusion Matrix Normalized — V10 Default (thr=0.5)",
        out_path=OUTPUT_DIR / "cm_v10_default.png",
    )
    plot_cm(
        CM_CALIBRATED,
        title="Confusion Matrix Normalized — V10 Calibrated (thr=0.170)",
        out_path=OUTPUT_DIR / "cm_v10_calibrated.png",
    )
    print(f"\nSelesai. File disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
