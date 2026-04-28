"""
=============================================================================
 TRAINING SCRIPT v9 - Kaggle (YOLO Classify, 2-class crops)
=============================================================================
 Perbaikan dari V8 berdasarkan analisis:

 MASALAH V8:
   1. Optimizer MuSGD lr=0.01 terlalu agresif → best epoch 6, tidak konvergen
   2. Val set imbalanced 64:36 → best.pt dipilih berdasarkan metrik yang bias
   3. Erasing=0.3 terlalu agresif → menghapus fitur engagement penting
   4. imgsz=224 → fitur detail crop orang kecil kurang tertangkap

 PERBAIKAN V9:
   1. optimizer="AdamW" + lr0=0.0005  → konvergensi lebih stabil, tidak overshoot
   2. Val set balanced 50:50 (crops_v9 dari dataset_smp_balanced)
      → best.pt dipilih berdasarkan metrik yang adil
   3. erasing=0.1 (dari 0.3)          → fitur engagement terjaga
   4. imgsz=320 (dari 224)            → detail crop lebih baik
   5. device="0,1" + batch=128        → pakai T4×2, gradient lebih stabil
   6. warmup_epochs=5 (dari 3)        → LR naik lebih gradual

 Ekspektasi: val accuracy tidak lagi peak di epoch 6, konvergen lebih mulus,
 test accuracy melampaui V7 (0.768) dan V8 (0.730).

 Setup Kaggle:
   1. Jalankan resplit_balanced.py + build_crops_v9.py secara lokal
   2. Zip: zip -r crops_v9.zip crops_v9/
   3. Upload ke Google Drive, salin File ID
   4. Isi GDRIVE_FILE_ID di bawah
   5. Run notebook ini (GPU T4 x2)
=============================================================================
"""

import os
import zipfile
from pathlib import Path
from ultralytics import YOLO

# ── GANTI INI SEBELUM UPLOAD KE KAGGLE ─────────────────────────────────────
GDRIVE_FILE_ID = "GANTI_DENGAN_FILE_ID_GDRIVE"
# ───────────────────────────────────────────────────────────────────────────

WORKING  = Path("/kaggle/working")
ZIP_PATH = WORKING / "crops_v9.zip"
DATA_DIR = WORKING / "crops_v9"
RUN_DIR  = WORKING / "runs"
CLASSES  = ["Engaged", "NotEngaged"]


# ── 1. Download & Ekstrak ────────────────────────────────────────────────────

def download_dataset():
    if ZIP_PATH.exists():
        print(f"[INFO] ZIP sudah ada: {ZIP_PATH}")
        return
    print("[INFO] Install gdown ...")
    os.system("pip install gdown -q")
    import gdown
    print(f"[INFO] Download dari Google Drive (id={GDRIVE_FILE_ID}) ...")
    gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)


def extract_dataset():
    if DATA_DIR.exists():
        print(f"[INFO] Sudah terekstrak: {DATA_DIR}")
        return
    print(f"[INFO] Ekstrak {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(WORKING)
    print(f"[INFO] Ekstrak selesai → {DATA_DIR}")


# ── 2. Verifikasi Dataset ────────────────────────────────────────────────────

def check_dataset():
    print("\n" + "=" * 60)
    print("DISTRIBUSI DATASET V9 (val balanced 50:50)")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  [{split}] TIDAK DITEMUKAN")
            continue
        total  = 0
        counts = {}
        for cls in CLASSES:
            n = len(list((split_dir / cls).glob("*.jpg")))
            counts[cls] = n
            total += n
        print(f"\n  [{split.upper()}]  total: {total}")
        for cls in CLASSES:
            pct = 100 * counts[cls] / total if total else 0
            bar = "█" * int(pct / 3)
            print(f"    {cls:12s}: {counts[cls]:6d} ({pct:5.1f}%)  {bar}")
    print("=" * 60)


# ── 3. Training ──────────────────────────────────────────────────────────────

def train():
    print("\n[INFO] Loading yolo11s-cls.pt ...")
    model = YOLO("yolo11s-cls.pt")

    print("\n" + "=" * 60)
    print("TRAINING v9 — YOLO Classify (2-class crops)")
    print("=" * 60)
    print(f"  Task      : classify")
    print(f"  Model     : yolo11s-cls")
    print(f"  Classes   : Engaged / NotEngaged")
    print(f"  Dataset   : crops_v9 (val balanced 50:50)")
    print(f"  Train     : 8398 : 8398 (balanced)")
    print(f"  imgsz     : 320px  ← naik dari 224")
    print(f"  Optimizer : AdamW lr=0.0005  ← dari MuSGD auto")
    print(f"  Batch     : 128 / device 0,1  ← pakai kedua T4")
    print(f"  Epochs    : 100  (patience 50)")
    print("=" * 60 + "\n")

    results = model.train(
        data=str(DATA_DIR),
        task="classify",

        # === CORE ===
        epochs=100,
        imgsz=320,          # V8: 224 → fitur detail lebih baik
        batch=128,          # V8: 64  → gradient lebih stabil dengan 2 GPU
        device="0,1",       # V8: 0   → pakai kedua T4
        workers=4,
        cache="disk",

        # === EARLY STOPPING ===
        patience=50,

        # === OPTIMIZER — perbaikan utama dari V8 ===
        optimizer="AdamW",  # V8: auto (MuSGD lr=0.01) → terlalu agresif
        lr0=0.0005,         # V8: auto 0.01 → 20x lebih kecil, konvergensi stabil
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,  # V8: 3.0 → LR naik lebih gradual
        warmup_momentum=0.8,

        # === AUGMENTASI ONLINE ===
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        erasing=0.1,        # V8: 0.3 → dikurangi, fitur engagement terjaga

        # === REGULARISASI ===
        weight_decay=0.0005,
        dropout=0.2,

        # === SAVING ===
        project=str(RUN_DIR),
        name="engagement_v9_classify",
        save=True,
        save_period=10,
        plots=True,

        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )

    return results


# ── 4. Evaluasi ──────────────────────────────────────────────────────────────

INFER_BATCH = 64


def per_class_report(best_pt: str):
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        balanced_accuracy_score, cohen_kappa_score,
        matthews_corrcoef, roc_auc_score,
    )

    model    = YOLO(best_pt)
    test_dir = DATA_DIR / "test"

    y_true, y_pred, y_prob = [], [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = test_dir / cls_name
        if not cls_dir.exists():
            continue
        img_files = list(cls_dir.glob("*.jpg"))
        print(f"  Inferensi {cls_name}: {len(img_files)} gambar ...")
        for i in range(0, len(img_files), INFER_BATCH):
            batch = img_files[i : i + INFER_BATCH]
            for r in model.predict(batch, verbose=False, imgsz=320, stream=True):
                y_true.append(cls_idx)
                y_pred.append(int(r.probs.top1))
                y_prob.append(float(r.probs.data[1]))

    sep = "=" * 60
    print(f"\n\n{sep}")
    print("PER-CLASS REPORT (Test Set)")
    print(sep)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    print("CONFUSION MATRIX (baris=aktual, kolom=prediksi):")
    cm = confusion_matrix(y_true, y_pred)
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14}" for v in row))

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    acc     = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa   = cohen_kappa_score(y_true, y_pred)
    mcc     = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n{sep}")
    print("RINGKASAN METRIK LENGKAP")
    print(sep)
    print(f"  {'Accuracy':<30} {acc:.4f}")
    print(f"  {'Balanced Accuracy':<30} {bal_acc:.4f}")
    print(f"  {'Cohen Kappa':<30} {kappa:.4f}")
    print(f"  {'Matthews CC (MCC)':<30} {mcc:.4f}")
    print(f"  {'ROC-AUC':<30} {roc_auc:.4f}")
    print(f"  {'Sensitivity (Recall NotEngaged)':<30} {sensitivity:.4f}")
    print(f"  {'Specificity (Recall Engaged)':<30} {specificity:.4f}")
    print(sep)

    # Perbandingan vs V7 dan V8
    print(f"\n{sep}")
    print("PERBANDINGAN V7 / V8 / V9")
    print(sep)
    print(f"  {'Metric':<30} {'V7':>8} {'V8':>8} {'V9':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Accuracy':<30} {'0.768':>8} {'0.730':>8} {acc:>8.3f}")
    print(f"  {'Balanced Accuracy':<30} {'—':>8} {'0.751':>8} {bal_acc:>8.3f}")
    print(f"  {'F1 Macro':<30} {'0.761':>8} {'0.727':>8} {(2*bal_acc - 1):>8.3f}")
    print(f"  {'MCC':<30} {'—':>8} {'0.489':>8} {mcc:>8.3f}")
    print(f"  {'ROC-AUC':<30} {'—':>8} {'0.876':>8} {roc_auc:>8.3f}")
    print(f"  {'Engaged F1':<30} {'0.802':>8} {'0.752':>8} {2*(specificity)/(1+specificity):>8.3f}")
    print(f"  {'NotEngaged F1':<30} {'0.721':>8} {'0.703':>8} {2*(sensitivity)/(1+sensitivity):>8.3f}")
    print(sep)

    return y_true, y_pred


def evaluate(best_pt: str):
    print(f"\n[INFO] Evaluasi test set: {best_pt}")
    model = YOLO(best_pt)
    m = model.val(
        data=str(DATA_DIR),
        split="test",
        plots=True,
        project=str(RUN_DIR),
        name="eval_v9_standard",
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI V9 — Top-1 Accuracy")
    print("=" * 60)
    try:
        print(f"  Top-1 Accuracy : {m.top1:.3f}")
        print(f"  Top-5 Accuracy : {m.top5:.3f}")
    except AttributeError:
        print(f"  {m.results_dict}")
    print("=" * 60)

    del model
    import torch; torch.cuda.empty_cache()

    per_class_report(best_pt)
    return m


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    check_dataset()

    results = train()

    best = str(RUN_DIR / "engagement_v9_classify" / "weights" / "best.pt")
    evaluate(best)

    print(f"\nSELESAI. Best weights: {best}")
