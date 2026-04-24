"""
Threshold tuning untuk V7 (YOLOv11s-cls, 2-class Engaged/NotEngaged).

Alur:
1. Kumpulkan probabilitas model di VAL set (crops_v7/valid)
2. Cari threshold optimal untuk kelas "Engaged"
   - Maksimalkan macro-F1 (primary)
   - Juga laporkan: Youden's J, balanced accuracy, accuracy
3. Terapkan threshold terbaik ke TEST set (crops_v7/test)
4. Bandingkan dengan baseline threshold 0.5

Output: classification report + confusion matrix di test set
        untuk baseline (0.5) dan threshold optimal.

Jalankan dari root project:
  python phase3_training/threshold_tuning_v7.py \
      --model models/best_v7.pt \
      --data phase2_dataset/crops_v7

Atau sesuaikan DEFAULT_MODEL / DEFAULT_DATA di bawah.
"""

from pathlib import Path
from collections import Counter
import argparse
import re
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

DEFAULT_MODEL = "models/best_v7.pt"
DEFAULT_DATA = "phase2_dataset/crops_v7"
CLASSES = ["Engaged", "NotEngaged"]  # index 0 = Engaged, 1 = NotEngaged
IMGSZ = 224
BATCH = 64


def collect_probs(model, split_dir: Path, return_paths=False):
    """Return (y_true, probs_engaged[, paths]) untuk semua gambar di split_dir."""
    y_true, p_engaged, paths = [], [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = split_dir / cls_name
        imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        print(f"  {cls_name}: {len(imgs)} gambar")
        for i in range(0, len(imgs), BATCH):
            batch = imgs[i : i + BATCH]
            for r, p in zip(
                model.predict(batch, verbose=False, imgsz=IMGSZ, stream=True), batch
            ):
                names = r.names
                data = r.probs.data.cpu().numpy()
                eng_idx = [k for k, v in names.items() if v == "Engaged"][0]
                p_engaged.append(float(data[eng_idx]))
                y_true.append(cls_idx)
                paths.append(p)
    if return_paths:
        return np.array(y_true), np.array(p_engaged), paths
    return np.array(y_true), np.array(p_engaged)


SESSION_PATTERNS = [
    re.compile(r"(Kelas\d+_[\d]+(?:[a-zA-Z]+)?_?[\d]*)"),  # mis. Kelas9_2mar_1012
    re.compile(r"(kelas\d+_[\d]+(?:[a-zA-Z]+)?_?[\d]*)", re.IGNORECASE),
]


def extract_session(filename: str) -> str:
    """Ekstrak nama sesi dari filename. Kalau tidak ketemu pola, pakai prefix sebelum frame number."""
    name = Path(filename).stem
    for pat in SESSION_PATTERNS:
        m = pat.search(name)
        if m:
            return m.group(1)
    # fallback: ambil prefix sebelum '_frame' atau '_aug' atau angka panjang terakhir
    base = re.split(r"_frame|_aug|_\d{4,}", name)[0]
    return base


def check_leakage(val_dir: Path, test_dir: Path):
    """Cek apakah ada nama sesi yang sama di val dan test."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIK A — Cek Session Leakage VAL vs TEST")
    print("=" * 60)
    val_sessions = Counter()
    test_sessions = Counter()
    for cls in CLASSES:
        for img in (val_dir / cls).glob("*.*"):
            val_sessions[extract_session(img.name)] += 1
        for img in (test_dir / cls).glob("*.*"):
            test_sessions[extract_session(img.name)] += 1

    overlap = set(val_sessions) & set(test_sessions)
    val_only = set(val_sessions) - set(test_sessions)
    test_only = set(test_sessions) - set(val_sessions)

    print(f"  Total sesi unik di VAL : {len(val_sessions)}")
    print(f"  Total sesi unik di TEST: {len(test_sessions)}")
    print(f"  Sesi OVERLAP val<->test: {len(overlap)}  <-- leakage indikator")
    print(f"  Sesi hanya di VAL      : {len(val_only)}")
    print(f"  Sesi hanya di TEST     : {len(test_only)}")
    if overlap:
        print(f"\n  Contoh sesi overlap (top 10 by count di test):")
        sorted_overlap = sorted(overlap, key=lambda s: -test_sessions[s])[:10]
        for s in sorted_overlap:
            print(f"    {s:40s} val={val_sessions[s]:4d}  test={test_sessions[s]:4d}")
    print(f"\n  Contoh nama file VAL  : {[Path(p).name for p in list((val_dir/'Engaged').glob('*.*'))[:3]]}")
    print(f"  Contoh nama file TEST : {[Path(p).name for p in list((test_dir/'Engaged').glob('*.*'))[:3]]}")
    return overlap, val_sessions, test_sessions


def search_best_threshold_on_test(y_test, p_test):
    """CHEATING — hanya untuk diagnosis ceiling V7. Jangan pakai untuk reporting."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIK B — Ceiling V7 (threshold dicari di TEST, cheating)")
    print("=" * 60)
    thrs = np.arange(0.10, 0.91, 0.01)
    best = None
    for t in thrs:
        y_pred = np.where(p_test >= t, 0, 1)
        acc = (y_pred == y_test).mean()
        mf1 = f1_score(y_test, y_pred, average="macro")
        bacc = balanced_accuracy_score(y_test, y_pred)
        if best is None or acc > best[1]:
            best = (t, acc, mf1, bacc)
    print(f"  Ceiling test V7 (max acc) : thr={best[0]:.3f} -> acc={best[1]:.4f} mF1={best[2]:.4f} bAcc={best[3]:.4f}")
    print(f"  Headroom dari baseline 0.5 : {(best[1] - (np.where(p_test>=0.5,0,1)==y_test).mean())*100:+.2f}%")
    print("  ^ Kalau headroom kecil, threshold tuning bukan kunci. Perlu retrain (V10).")
    return best


def eval_at_threshold(y_true, p_engaged, thr, label):
    # pred = 0 (Engaged) jika p_engaged >= thr, else 1 (NotEngaged)
    y_pred = np.where(p_engaged >= thr, 0, 1)
    print(f"\n{'=' * 60}")
    print(f"{label}  (threshold Engaged >= {thr:.3f})")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
    print("Confusion matrix (baris=aktual, kolom=prediksi):")
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14}" for v in row))
    acc = (y_pred == y_true).mean()
    mf1 = f1_score(y_true, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)
    print(f"Top-1 Acc: {acc:.4f} | Macro-F1: {mf1:.4f} | Balanced Acc: {bacc:.4f}")
    return acc, mf1, bacc


def search_best_threshold(y_true, p_engaged):
    """Grid search threshold Engaged di [0.1, 0.9] step 0.01."""
    thrs = np.arange(0.10, 0.91, 0.01)
    rows = []
    for t in thrs:
        y_pred = np.where(p_engaged >= t, 0, 1)
        acc = (y_pred == y_true).mean()
        mf1 = f1_score(y_true, y_pred, average="macro")
        bacc = balanced_accuracy_score(y_true, y_pred)
        # Youden's J untuk kelas Engaged
        tp = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 1) & (y_true == 0)).sum()
        fp = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 1) & (y_true == 1)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        youden = tpr - fpr
        rows.append((t, acc, mf1, bacc, youden))
    arr = np.array(rows)
    best_mf1 = arr[arr[:, 2].argmax()]
    best_acc = arr[arr[:, 1].argmax()]
    best_bacc = arr[arr[:, 3].argmax()]
    best_you = arr[arr[:, 4].argmax()]
    print("\nTop kandidat threshold di VAL:")
    print(f"  Max Macro-F1      : thr={best_mf1[0]:.3f} -> acc={best_mf1[1]:.4f} mF1={best_mf1[2]:.4f} bAcc={best_mf1[3]:.4f}")
    print(f"  Max Accuracy      : thr={best_acc[0]:.3f} -> acc={best_acc[1]:.4f} mF1={best_acc[2]:.4f} bAcc={best_acc[3]:.4f}")
    print(f"  Max Balanced Acc  : thr={best_bacc[0]:.3f} -> acc={best_bacc[1]:.4f} mF1={best_bacc[2]:.4f} bAcc={best_bacc[3]:.4f}")
    print(f"  Max Youden's J    : thr={best_you[0]:.3f} -> acc={best_you[1]:.4f} mF1={best_you[2]:.4f} J={best_you[4]:.4f}")
    return {
        "macro_f1": float(best_mf1[0]),
        "accuracy": float(best_acc[0]),
        "balanced_acc": float(best_bacc[0]),
        "youden": float(best_you[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    model = YOLO(args.model)
    print(f"Model classes: {model.names}")
    data = Path(args.data)

    # Diagnostik A — leakage (tidak butuh inferensi, jalankan duluan)
    check_leakage(data / "valid", data / "test")

    print("\n[1/3] Mengumpulkan probabilitas di VAL ...")
    y_val, p_val = collect_probs(model, data / "valid")
    try:
        print(f"  ROC-AUC (val): {roc_auc_score(y_val == 0, p_val):.4f}")
    except Exception:
        pass

    print("\n[2/3] Mencari threshold optimal di VAL ...")
    best = search_best_threshold(y_val, p_val)

    print("\n[3/3] Evaluasi di TEST ...")
    y_test, p_test = collect_probs(model, data / "test")
    try:
        print(f"  ROC-AUC (test): {roc_auc_score(y_test == 0, p_test):.4f}")
    except Exception:
        pass

    # Diagnostik B — ceiling V7
    search_best_threshold_on_test(y_test, p_test)

    # Baseline
    eval_at_threshold(y_test, p_test, 0.5, "TEST — baseline (thr=0.5)")
    # Best macro-F1 (primary)
    eval_at_threshold(
        y_test, p_test, best["macro_f1"], f"TEST — tuned (Max Macro-F1 di val)"
    )
    # Best balanced accuracy (sering paling generalize)
    if abs(best["balanced_acc"] - best["macro_f1"]) > 1e-6:
        eval_at_threshold(
            y_test, p_test, best["balanced_acc"], f"TEST — tuned (Max Balanced-Acc di val)"
        )
    # Best Youden
    if abs(best["youden"] - best["macro_f1"]) > 1e-6 and abs(best["youden"] - best["balanced_acc"]) > 1e-6:
        eval_at_threshold(
            y_test, p_test, best["youden"], f"TEST — tuned (Max Youden di val)"
        )


if __name__ == "__main__":
    main()
