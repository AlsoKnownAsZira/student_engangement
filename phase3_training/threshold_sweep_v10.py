"""
Threshold sweep V10 — menampilkan seluruh hasil grid search (bukan cuma threshold terbaik).

Mereproduksi calibration set yang sama persis dengan calibrate_v10.py (seed=42,
calib_frac=0.20, stratified split) lalu menghitung Accuracy/Macro-F1/Balanced-Acc
di SETIAP titik threshold 0.05-0.95 (step 0.01), menyimpan ke CSV dan membuat
grafik garis Macro-F1 vs Threshold dengan titik 0.170 ditandai.

Jalankan:
  python phase3_training/threshold_sweep_v10.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import f1_score, balanced_accuracy_score

CLASSES = ["Engaged", "NotEngaged"]
IMGSZ = 224
BATCH = 64
CALIB_FRAC = 0.20
SEED = 42
SELECTED_THR = 0.170

_root = Path(__file__).parent.parent
MODEL_PATH = _root / "models" / "best_v10.pt"
DATA_DIR = _root / "phase2_dataset" / "crops_v10"
OUT_DIR = _root / "phase3_training" / "outputs"


def collect_probs(model, split_dir: Path):
    y_true, p_eng = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = split_dir / cls_name
        files = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        print(f"  {cls_name}: {len(files)} gambar")
        for i in range(0, len(files), BATCH):
            batch = files[i : i + BATCH]
            for r in model.predict(batch, verbose=False, imgsz=IMGSZ, stream=True):
                names = r.names
                eng_idx = [k for k, v in names.items() if v == "Engaged"][0]
                p_eng.append(float(r.probs.data.cpu().numpy()[eng_idx]))
                y_true.append(cls_idx)
    return np.array(y_true), np.array(p_eng)


def stratified_split(y_true, calib_frac, seed):
    rng = np.random.default_rng(seed)
    idx_calib, idx_final = [], []
    for cls_idx in range(len(CLASSES)):
        cls_indices = np.where(y_true == cls_idx)[0]
        rng.shuffle(cls_indices)
        n_calib = int(round(len(cls_indices) * calib_frac))
        idx_calib.extend(cls_indices[:n_calib].tolist())
        idx_final.extend(cls_indices[n_calib:].tolist())
    return np.array(sorted(idx_calib)), np.array(sorted(idx_final))


def sweep(y, p):
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        t = round(float(t), 2)
        y_pred = np.where(p >= t, 0, 1)
        acc = float((y_pred == y).mean())
        mf1 = float(f1_score(y, y_pred, average="macro"))
        bacc = float(balanced_accuracy_score(y, y_pred))
        rows.append((t, acc, mf1, bacc))
    return rows


def main():
    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_DIR}/test")
    model = YOLO(str(MODEL_PATH))
    test_dir = DATA_DIR / "test"

    print("\n[1/3] Mengumpulkan probabilitas di TEST set ...")
    y_test, p_test = collect_probs(model, test_dir)
    print(f"  Total: {len(y_test)} sampel")

    print(f"\n[2/3] Reproduksi calibration split (seed={SEED}, calib_frac={CALIB_FRAC}) ...")
    idx_calib, _ = stratified_split(y_test, CALIB_FRAC, SEED)
    y_calib, p_calib = y_test[idx_calib], p_test[idx_calib]
    print(f"  Calibration set: {len(y_calib)} sampel")

    print("\n[3/3] Grid search 0.05-0.95 (step 0.01) di calibration set ...")
    rows = sweep(y_calib, p_calib)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "threshold_sweep_v10.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("threshold,accuracy,macro_f1,balanced_acc\n")
        for t, acc, mf1, bacc in rows:
            f.write(f"{t:.2f},{acc:.4f},{mf1:.4f},{bacc:.4f}\n")
    print(f"\nCSV disimpan di: {csv_path}")

    best_row = max(rows, key=lambda r: r[2])
    print(f"Threshold terbaik (Macro-F1 max): {best_row[0]:.3f} -> Macro-F1={best_row[2]:.4f}")

    thresholds = [r[0] for r in rows]
    macro_f1s = [r[2] for r in rows]
    accs = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, macro_f1s, color="#1f77b4", lw=2, label="Macro F1-Score")
    ax.plot(thresholds, accs, color="#ff7f0e", lw=1.5, linestyle="--", label="Accuracy", alpha=0.8)

    sel_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - SELECTED_THR))
    ax.scatter([thresholds[sel_idx]], [macro_f1s[sel_idx]], color="red", zorder=5, s=60,
               label=f"Threshold terpilih ({SELECTED_THR:.3f})")
    ax.annotate(f"({SELECTED_THR:.3f}, {macro_f1s[sel_idx]:.3f})",
                (thresholds[sel_idx], macro_f1s[sel_idx]),
                textcoords="offset points", xytext=(10, -12), fontsize=9, color="red")

    ax.set_xlabel("Threshold (Engaged >= t)", fontsize=11)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title("Grid Search Threshold — Calibration Set (V10)", fontsize=12)
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    fig.tight_layout()

    png_path = OUT_DIR / "threshold_sweep_v10.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafik disimpan di: {png_path}")


if __name__ == "__main__":
    main()
