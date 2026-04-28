"""
=============================================================================
 TRAINING SCRIPT v7 - Kaggle (YOLO Classify, 2-class crops)
=============================================================================
 Perubahan vs v6:
   - 2 class: Engaged (High+Medium) vs NotEngaged (Low)
   - Train: perfectly balanced 8353:8353
   - Augment offline HANYA Low (bukan semua kelas) → High dan Medium
     tidak diperbanyak artifisial, sehingga sinyal aslinya terjaga
   - Augmentasi online tetap aktif untuk variasi tambahan

 Alasan kembali ke 2-class:
   - v6 (3-class crop) Medium F1 hanya 0.335 — ambiguitas visual intrinsik
   - v5 (2-class detect) mAP 76.2% membuktikan merge High+Medium bekerja
   - v7 = kekuatan v5 (2-class) + kekuatan v6 (crop per orang, no bbox error)

 Setup Kaggle:
   1. Zip folder crops_v7/ secara lokal
   2. Upload ke Google Drive, salin File ID
   3. Isi GDRIVE_FILE_ID di bawah
   4. Run notebook ini (GPU T4 x2 atau P100)
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
ZIP_PATH = WORKING / "crops_v7.zip"
DATA_DIR = WORKING / "crops_v7"
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
    print("DISTRIBUSI DATASET V7 (crop 2-class)")
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
    print("TRAINING v7 — YOLO Classify (2-class crops)")
    print("=" * 55)
    print(f"  Task    : classify")
    print(f"  Model   : yolo11s-cls")
    print(f"  Classes : Engaged / NotEngaged")
    print(f"  Train   : 8353 : 8353 (perfectly balanced)")
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
        name="engagement_v7_classify",
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
    from sklearn.metrics import classification_report, confusion_matrix

    model = YOLO(best_pt)
    test_dir = DATA_DIR / "test"

    y_true, y_pred = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = test_dir / cls_name
        if not cls_dir.exists():
            continue
        img_files = list(cls_dir.glob("*.jpg"))
        print(f"  Inferensi {cls_name}: {len(img_files)} gambar ...")
        # stream=True + batch manual → hindari OOM saat list besar
        for i in range(0, len(img_files), INFER_BATCH):
            batch = img_files[i : i + INFER_BATCH]
            for r in model.predict(batch, verbose=False, imgsz=224, stream=True):
                y_true.append(cls_idx)
                y_pred.append(int(r.probs.top1))

    print("\n")
    print("=" * 60)
    print("PER-CLASS REPORT (Test Set)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    print("CONFUSION MATRIX (baris=aktual, kolom=prediksi):")
    cm = confusion_matrix(y_true, y_pred)
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14}" for v in row))
    print("=" * 60)

    overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    print(f"Overall Top-1 Accuracy: {overall_acc:.3f}")
    return y_true, y_pred


def evaluate(best_pt: str):
    print(f"\n[INFO] Evaluasi test set: {best_pt}")
    model = YOLO(best_pt)

    print("\n--- Standard ---")
    m = model.val(
        data=str(DATA_DIR),
        split="test",
        plots=True,
        project=str(RUN_DIR),
        name="eval_v7_standard",
    )

    print("\n--- TTA ---")
    m_tta = model.val(
        data=str(DATA_DIR),
        split="test",
        augment=True,
        plots=True,
        project=str(RUN_DIR),
        name="eval_v7_tta",
    )

    print("\n" + "=" * 55)
    print("HASIL EVALUASI V7 — Top-1 Accuracy")
    print("=" * 55)
    print(f"{'Metric':<25} {'Standard':>10} {'TTA':>10}")
    print("-" * 47)
    try:
        print(f"{'Top-1 Accuracy':<25} {m.top1:>10.3f} {m_tta.top1:>10.3f}")
    except AttributeError:
        print(f"  Standard: {m.results_dict}")
        print(f"  TTA     : {m_tta.results_dict}")
    print("=" * 55)

    per_class_report(best_pt)
    return m, m_tta


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    check_dataset()

    results = train()

    best = str(RUN_DIR / "engagement_v7_classify" / "weights" / "best.pt")
    evaluate(best)

    print(f"\nSELESAI. Best weights: {best}")
