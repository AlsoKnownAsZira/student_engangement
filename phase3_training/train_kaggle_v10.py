"""
=============================================================================
 TRAINING SCRIPT v10 - Kaggle (YOLO Classify, 2-class crops)
=============================================================================
 Strategi V10 (target: test acc >= 80%):

 1. SESSION-BASED MULTI-VAL SPLIT
    - Train  : 4 sesi (2mar_0830, 3mar_1102, 5mar_0824, 5mar_1024)
    - Val    : 2 sesi (4mar_0917 + 6mar_0959) <-- multi-session, fix V7 bug
    - Test   : 1 sesi (2mar_0906) <-- sama V7, apple-to-apple

 2. NO OFFLINE AUGMENTATION
    - Semua file `_aug` di crops_v7 di-DROP saat build crops_v10
    - Train rasio asli: ~70:30 (Engaged:NotEngaged)

 3. CLASS IMBALANCE via OVERSAMPLE (symlink minority)
    - NotEngaged (minority) di-symlink-duplikat sampai mendekati Engaged
    - Tiap epoch online aug (fliplr, hsv, scale, erasing) bikin kopian
      kelihatan beda, jadi bukan persis sama dengan offline aug
    - Default: oversample factor dihitung otomatis di build_crops_v10

 4. TRAINING CONFIG (mirip V7 yang terbukti, tweaked)
    - Optimizer: SGD lr0=0.001, warmup 5, cos_lr
    - Patience 30 (lebih sabar dari V7)
    - Online aug aktif, dropout 0.2
    - imgsz 224

 5. POST-TRAINING
    - Threshold tuning di val baru (multi-session) -> pilih thr optimal
    - Evaluasi test dengan thr default 0.5 DAN thr tuned
    - TTA juga dievaluasi

 Setup Kaggle:
   1. Jalankan `python phase2_dataset/build_crops_v10.py` lokal dulu
   2. Zip crops_v10/, upload ke GDrive, salin File ID
   3. Isi GDRIVE_FILE_ID di bawah
   4. Run di Kaggle (T4 x2 / P100)
=============================================================================
"""

import os
import zipfile
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ── GANTI INI SEBELUM UPLOAD KE KAGGLE ─────────────────────────────────────
GDRIVE_FILE_ID = "GANTI_DENGAN_FILE_ID_GDRIVE"
# ───────────────────────────────────────────────────────────────────────────

WORKING  = Path("/kaggle/working")
ZIP_PATH = WORKING / "crops_v10.zip"
DATA_DIR = WORKING / "crops_v10"
RUN_DIR  = WORKING / "runs"
CLASSES  = ["Engaged", "NotEngaged"]


# ── 1. Download & Ekstrak ────────────────────────────────────────────────────

def download_dataset():
    if ZIP_PATH.exists():
        print(f"[INFO] ZIP sudah ada: {ZIP_PATH}")
        return
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


# ── 2. Oversample minority class via duplikasi file di TRAIN ─────────────────

def oversample_train_minority():
    """Duplikasi file NotEngaged di train sampai jumlah ~ Engaged.
    Menggunakan copy biasa (Kaggle filesystem kadang restrict symlink).
    Hanya jalan kalau train belum dibalance.
    """
    import shutil
    train_eng = DATA_DIR / "train" / "Engaged"
    train_low = DATA_DIR / "train" / "NotEngaged"
    n_eng = len(list(train_eng.glob("*.jpg")))
    n_low = len(list(train_low.glob("*.jpg")))
    print(f"[INFO] Train asli: Engaged={n_eng}, NotEngaged={n_low}")

    if n_low >= n_eng * 0.95:
        print("[INFO] Sudah seimbang, skip oversample.")
        return

    # Berapa kopian total NotEng yang dibutuhkan
    target = n_eng
    factor = target / n_low
    print(f"[INFO] Oversample NotEngaged x{factor:.2f} -> {target}")

    files = sorted(train_low.glob("*.jpg"))
    full_rounds = int(factor) - 1   # selain original yang sudah ada
    remainder = target - n_low * (full_rounds + 1)

    copied = 0
    for r in range(full_rounds):
        for f in files:
            new_name = f.stem + f"__dup{r+1}" + f.suffix
            dst = train_low / new_name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
    # Sisa parsial
    for i in range(remainder):
        f = files[i % len(files)]
        new_name = f.stem + f"__dup{full_rounds+1}_{i}" + f.suffix
        dst = train_low / new_name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied += 1

    n_low_after = len(list(train_low.glob("*.jpg")))
    print(f"[INFO] Selesai oversample. NotEngaged sekarang: {n_low_after} (+{copied})")


# ── 3. Verifikasi Dataset ────────────────────────────────────────────────────

def check_dataset():
    print("\n" + "=" * 55)
    print("DISTRIBUSI DATASET V10 (crop 2-class, multi-session val)")
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


# ── 4. Training ──────────────────────────────────────────────────────────────

def train():
    print("\n[INFO] Loading yolo11s-cls.pt ...")
    model = YOLO("yolo11s-cls.pt")

    print("\n" + "=" * 55)
    print("TRAINING v10 — YOLO Classify (2-class, session-based split)")
    print("=" * 55)

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
        patience=30,

        # === LEARNING RATE (SGD) ===
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        momentum=0.937,

        # === AUGMENTASI ONLINE (lebih kuat untuk variasi NotEng yang dioversample) ===
        fliplr=0.5,
        flipud=0.0,
        degrees=12.0,
        translate=0.12,
        scale=0.45,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.5,
        erasing=0.35,

        # === REGULARISASI ===
        weight_decay=0.0005,
        dropout=0.2,
        label_smoothing=0.05,

        # === SAVING ===
        project=str(RUN_DIR),
        name="engagement_v10_classify",
        save=True,
        save_period=10,
        plots=True,

        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )

    return results


# ── 5. Threshold tuning di val + evaluasi test ───────────────────────────────

INFER_BATCH = 64


def collect_probs(model, split_dir: Path):
    y_true, p_eng = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        for i in range(0, len(files), INFER_BATCH):
            batch = files[i : i + INFER_BATCH]
            for r in model.predict(batch, verbose=False, imgsz=224, stream=True):
                names = r.names
                eng_idx = [k for k, v in names.items() if v == "Engaged"][0]
                p_eng.append(float(r.probs.data.cpu().numpy()[eng_idx]))
                y_true.append(cls_idx)
    return np.array(y_true), np.array(p_eng)


def find_best_threshold(y_val, p_val):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    best = (0.5, -1, -1, -1)
    for t in np.arange(0.10, 0.91, 0.01):
        y_pred = np.where(p_val >= t, 0, 1)
        mf1 = f1_score(y_val, y_pred, average="macro")
        acc = (y_pred == y_val).mean()
        bacc = balanced_accuracy_score(y_val, y_pred)
        if mf1 > best[2]:
            best = (float(t), float(acc), float(mf1), float(bacc))
    return best


def report_at(y_test, p_test, thr, label):
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = np.where(p_test >= thr, 0, 1)
    print(f"\n{'=' * 60}\n{label}  (thr Engaged>={thr:.3f})\n{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
    print("Confusion matrix:")
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14}" for v in row))
    acc = (y_pred == y_test).mean()
    print(f"Top-1 Acc: {acc:.4f}")
    return acc


def evaluate(best_pt: str):
    from sklearn.metrics import roc_auc_score
    model = YOLO(best_pt)

    # 1. Standard val.test (default thr=0.5)
    print("\n[INFO] Ultralytics val (standard) ...")
    m = model.val(
        data=str(DATA_DIR), split="test", plots=True,
        project=str(RUN_DIR), name="eval_v10_standard",
    )
    print("\n[INFO] Ultralytics val (TTA) ...")
    m_tta = model.val(
        data=str(DATA_DIR), split="test", augment=True, plots=True,
        project=str(RUN_DIR), name="eval_v10_tta",
    )
    try:
        print(f"  Top-1 standard: {m.top1:.4f}  |  TTA: {m_tta.top1:.4f}")
    except AttributeError:
        pass

    # 2. Threshold tuning di val
    print("\n[INFO] Mengumpulkan probs val (untuk threshold tuning) ...")
    y_val, p_val = collect_probs(model, DATA_DIR / "valid")
    print(f"  ROC-AUC val : {roc_auc_score(y_val == 0, p_val):.4f}")

    print("\n[INFO] Mengumpulkan probs test ...")
    y_test, p_test = collect_probs(model, DATA_DIR / "test")
    print(f"  ROC-AUC test: {roc_auc_score(y_test == 0, p_test):.4f}")

    thr, vacc, vmf1, vbacc = find_best_threshold(y_val, p_val)
    print(f"\n[INFO] Best thr di val (max macro-F1): thr={thr:.3f} "
          f"-> val_acc={vacc:.4f} mF1={vmf1:.4f} bAcc={vbacc:.4f}")

    # 3. Report di test pada baseline & tuned
    acc_base = report_at(y_test, p_test, 0.50, "TEST — baseline (thr=0.5)")
    acc_tuned = report_at(y_test, p_test, thr, f"TEST — tuned (thr={thr:.3f})")

    print("\n" + "=" * 60)
    print("RINGKASAN V10")
    print("=" * 60)
    print(f"  Baseline (thr=0.5)     : {acc_base:.4f}")
    print(f"  Tuned    (thr={thr:.3f}) : {acc_tuned:.4f}")
    try:
        print(f"  Ultralytics TTA        : {m_tta.top1:.4f}")
    except AttributeError:
        pass
    print(f"  V7 reference (test)    : 0.7681")
    print(f"  Target                 : >= 0.8000")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    oversample_train_minority()
    check_dataset()

    results = train()

    best = str(RUN_DIR / "engagement_v10_classify" / "weights" / "best.pt")
    evaluate(best)

    print(f"\nSELESAI. Best weights: {best}")
