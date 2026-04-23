"""
=============================================================================
 TRAINING SCRIPT v8 - Kaggle (YOLO Classify, 2-class crops)
=============================================================================
 Perubahan vs v7:
   - Dataset baru (dataset_smp, versi 13 Roboflow) dengan Low lebih banyak:
       train → High 4987 | Medium 3411 | Low 6130
   - Mapping tetap: High+Medium → Engaged | Low → NotEngaged
   - Augmentasi offline lebih ringan (~37% synthetic vs v7 ~115%):
       Low 6130 → augment ke ~8398 (selisih ~2268 saja)
   - Augmentasi offline lebih ringan (±20% brightness/contrast vs v7 ±30%)
   - Hyperparameter sama dengan v7 (sudah terbukti stabil)

 Setup Kaggle:
   1. Jalankan build_crops_v8.py secara lokal → crops_v8/
   2. Zip: zip -r crops_v8.zip crops_v8/
   3. Upload ke Google Drive, salin File ID
   4. Isi GDRIVE_FILE_ID di bawah
   5. Run notebook ini (GPU T4 x2 atau P100)
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
ZIP_PATH = WORKING / "crops_v8.zip"
DATA_DIR = WORKING / "crops_v8"
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
    print("\n" + "=" * 55)
    print("DISTRIBUSI DATASET V8 (crop 2-class, dataset_smp)")
    print("=" * 55)
    for split in ["train", "valid", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  [{split}] TIDAK DITEMUKAN")
            continue
        total = 0
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
    print("=" * 55)


# ── 3. Training ──────────────────────────────────────────────────────────────

def train():
    print("\n[INFO] Loading yolo11s-cls.pt ...")
    model = YOLO("yolo11s-cls.pt")

    print("\n" + "=" * 55)
    print("TRAINING v8 — YOLO Classify (2-class crops)")
    print("=" * 55)
    print(f"  Task    : classify")
    print(f"  Model   : yolo11s-cls")
    print(f"  Classes : Engaged / NotEngaged")
    print(f"  Dataset : dataset_smp (Low lebih banyak, aug ringan ~37%)")
    print(f"  Train   : ~8398 : ~8398 (balanced)")
    print(f"  imgsz   : 224px")
    print(f"  Epochs  : 100  (patience 50)")
    print(f"  Batch   : 64")
    print("=" * 55 + "\n")

    results = model.train(
        data=str(DATA_DIR),
        task="classify",

        # === CORE ===
        epochs=100,
        imgsz=224,
        batch=64,
        device=0,
        workers=4,
        cache="disk",

        # === EARLY STOPPING ===
        patience=50,

        # === LEARNING RATE ===
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
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
        erasing=0.3,

        # === REGULARISASI ===
        weight_decay=0.0005,
        dropout=0.2,

        # === SAVING ===
        project=str(RUN_DIR),
        name="engagement_v8_classify",
        save=True,
        save_period=10,
        plots=True,

        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )

    return results


# ── 4. Evaluasi ──────────────────────────────────────────────────────────────

INFER_BATCH = 64  # kecil agar tidak OOM di T4


def per_class_report(best_pt: str):
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        balanced_accuracy_score, cohen_kappa_score,
        matthews_corrcoef, roc_auc_score,
    )

    model = YOLO(best_pt)
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
            for r in model.predict(batch, verbose=False, imgsz=224, stream=True):
                y_true.append(cls_idx)
                y_pred.append(int(r.probs.top1))
                y_prob.append(float(r.probs.data[1]))  # prob kelas index-1 (NotEngaged)

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

    # Turunkan TP/FP/FN/TN dari confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)   # Recall NotEngaged
    specificity = tn / (tn + fp)   # Recall Engaged

    acc          = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    kappa        = cohen_kappa_score(y_true, y_pred)
    mcc          = matthews_corrcoef(y_true, y_pred)
    roc_auc      = roc_auc_score(y_true, y_prob)

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

    return y_true, y_pred


def evaluate(best_pt: str):
    print(f"\n[INFO] Evaluasi test set: {best_pt}")

    # TTA (augment=True) tidak didukung ClassificationModel → OOM + spam warning
    # Hanya jalankan standard eval
    model = YOLO(best_pt)
    m = model.val(
        data=str(DATA_DIR),
        split="test",
        plots=True,
        project=str(RUN_DIR),
        name="eval_v8_standard",
    )

    print("\n" + "=" * 55)
    print("HASIL EVALUASI V8 — Top-1 Accuracy")
    print("=" * 55)
    try:
        print(f"  Top-1 Accuracy : {m.top1:.3f}")
        print(f"  Top-5 Accuracy : {m.top5:.3f}")
    except AttributeError:
        print(f"  {m.results_dict}")
    print("=" * 55)

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

    best = str(RUN_DIR / "engagement_v8_classify" / "weights" / "best.pt")
    evaluate(best)

    print(f"\nSELESAI. Best weights: {best}")
