"""
=============================================================================
 TRAINING SCRIPT v4 - Google Colab (Tesla T4)
=============================================================================
 Perubahan kunci dari v2:
   1. Dataset V4 — hasil targeted Medium augmentation (augment_medium.py)
      Medium instances naik dari 3366 → ~5000+ (mendekati proporsi High)
   2. cls loss weight 1.5 (dari 1.0) → lebih menekan klasifikasi Medium
   3. copy_paste 0.3 (dari 0.1) → lebih agresif paste augmentasi
   4. Semua improvement V2 dipertahankan

 Hipotesis:
   Plateau ~58% disebabkan oleh class imbalance (Medium 27% data).
   Dengan menyeimbangkan Medium ke ~35%, model diharapkan bisa naik ke 63-67%.

=============================================================================
 USAGE (di Google Colab):
   1. Jalankan augment_medium.py di lokal terlebih dahulu:
      python phase3_training/augment_medium.py --target 2000
   2. Zip folder dataset_yolo_split (termasuk train_v4/):
      → Pastikan train_v4/images/ dan train_v4/labels/ ikut ter-zip
   3. Upload dataset_yolo_v4.zip ke Colab
   4. Jalankan script ini
=============================================================================
"""

from ultralytics import YOLO
import os
import shutil


def setup_dataset_v4(zip_path="/content/dataset_yolo_v4.zip", extract_to="/content"):
    """Ekstrak dataset V4 dan update path di data.yaml"""
    import zipfile

    dataset_dir = os.path.join(extract_to, "dataset_yolo_split")

    if not os.path.exists(dataset_dir):
        print(f"[INFO] Mengekstrak dataset dari {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[INFO] Ekstraksi selesai!")
    else:
        print("[INFO] Dataset sudah terekstrak, skip.")

    # Update data.yaml — pakai train_v4 sebagai train set
    yaml_path = os.path.join(dataset_dir, "data_v4.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train_v4/images\n")   # <-- V4 augmented train set
        f.write("val:   valid/images\n")
        f.write("test:  test/images\n")
        f.write("\n")
        f.write("nc: 3\n")
        f.write("names: ['High', 'Low', 'Medium']\n")

    print(f"[INFO] data_v4.yaml updated: train = train_v4/images")
    return yaml_path


def count_dataset(dataset_dir="/content/dataset_yolo_split", train_split="train_v4"):
    """Hitung distribusi kelas di dataset V4"""
    splits = [train_split, 'valid', 'test']
    class_names = {0: 'High', 1: 'Low', 2: 'Medium'}

    print("\n" + "=" * 60)
    print("DISTRIBUSI DATASET V4")
    print("=" * 60)

    for split in splits:
        label_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(label_dir):
            print(f"  [{split.upper()}] tidak ditemukan, skip.")
            continue

        counts = {0: 0, 1: 0, 2: 0}
        n_images = 0

        for fname in os.listdir(label_dir):
            if fname.endswith('.txt'):
                n_images += 1
                with open(os.path.join(label_dir, fname)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            if cls in counts:
                                counts[cls] += 1

        total = sum(counts.values())
        print(f"\n  [{split.upper()}] {n_images} images, {total} annotations")
        for cls_id, name in class_names.items():
            pct = counts[cls_id] / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {name:8s}: {counts[cls_id]:5d} ({pct:5.1f}%) {bar}")

    print("=" * 60)


def train_v4(dataset_yaml, model_size="yolo11m.pt"):
    """
    Training YOLO11m dengan dataset V4 (Medium-balanced).

    Perubahan dari V2:
    - Dataset pakai train_v4 (Medium sudah diaugmentasi)
    - cls=1.5 (naik dari 1.0) → lebih fokus ke klasifikasi
    - copy_paste=0.3 (naik dari 0.1) → lebih agresif online augmentation
    """
    print(f"\n[INFO] Loading pre-trained model: {model_size}")
    model = YOLO(model_size)

    print("\n" + "=" * 60)
    print("MEMULAI TRAINING v4 - MEDIUM BALANCED DATASET")
    print("=" * 60)
    print(f"  Model       : {model_size}")
    print(f"  Resolution  : 960px")
    print(f"  Dataset     : train_v4 (Medium augmented)")
    print(f"  cls weight  : 1.5 (naik dari 1.0)")
    print(f"  copy_paste  : 0.3 (naik dari 0.1)")
    print(f"  patience    : 70  (naik dari 40)")
    print(f"  label_smooth: 0.1 (baru)")
    print(f"  close_mosaic: 30  (naik dari 20)")
    print("=" * 60)

    results = model.train(
        data=dataset_yaml,

        # === CORE ===
        epochs=150,
        imgsz=960,
        batch=6,
        device=0,
        workers=4,

        # === EARLY STOPPING ===
        patience=70,          # naik dari 40 → dataset 2.7x lebih besar, butuh waktu konvergen lebih lama

        # === LEARNING RATE ===
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # === LOSS WEIGHTS ===
        box=7.5,
        cls=1.5,           # naik dari 1.0 → lebih tegas bedakan Medium
        dfl=1.5,

        # === AUGMENTASI ONLINE ===
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,    # kembali ke default → imbalance sudah fix via offline augment
        erasing=0.3,

        degrees=15.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,

        fliplr=0.5,
        flipud=0.0,

        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,

        close_mosaic=30,       # naik dari 20 → beri waktu lebih lama tanpa augmentasi berat

        # === REGULARISASI ===
        weight_decay=0.001,
        dropout=0.1,
        label_smoothing=0.1,   # bantu ambiguitas batas High/Medium/Low yang subjektif

        # === SAVING ===
        name="student_engagement_v4",
        save=True,
        save_period=10,
        plots=True,

        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING SELESAI!")
    print("=" * 60)
    print("Model terbaik: runs/detect/student_engagement_v4/weights/best.pt")
    print("Model terakhir: runs/detect/student_engagement_v4/weights/last.pt")

    return results


def evaluate_on_test(model_path, dataset_yaml):
    """Evaluasi model V4 pada test set"""
    print(f"\n[INFO] Evaluasi model V4: {model_path}")
    model = YOLO(model_path)

    print("\nEvaluasi STANDARD (tanpa TTA):")
    metrics_standard = model.val(
        data=dataset_yaml,
        split='test',
        plots=True,
        save_json=True,
        name="eval_v4_standard"
    )

    print("\nEvaluasi dengan TTA:")
    metrics_tta = model.val(
        data=dataset_yaml,
        split='test',
        augment=True,
        plots=True,
        save_json=True,
        name="eval_v4_tta"
    )

    print("\n" + "=" * 60)
    print("PERBANDINGAN HASIL V4")
    print("=" * 60)
    print(f"{'Metric':<20} {'Standard':>10} {'TTA':>10}")
    print("-" * 42)
    print(f"{'mAP@50':<20} {metrics_standard.box.map50:>10.3f} {metrics_tta.box.map50:>10.3f}")
    print(f"{'mAP@50-95':<20} {metrics_standard.box.map:>10.3f} {metrics_tta.box.map:>10.3f}")

    # Per-class breakdown
    print("\nPer-class mAP@50:")
    class_names = ['High', 'Low', 'Medium']
    for i, name in enumerate(class_names):
        try:
            ap = metrics_standard.box.ap50[i]
            print(f"  {name:8s}: {ap:.3f}")
        except Exception:
            pass
    print("=" * 60)

    return metrics_standard, metrics_tta


# =============================================================================
# MAIN EXECUTION (jalankan di Colab)
# =============================================================================
if __name__ == "__main__":
    # Step 1: Setup dataset V4
    yaml_path = setup_dataset_v4()

    # Step 2: Lihat distribusi dataset V4
    count_dataset()

    # Step 3: Training!
    results = train_v4(yaml_path, model_size="yolo11m.pt")

    # Step 4: Evaluasi di test set
    best_model = "/content/runs/detect/student_engagement_v4/weights/best.pt"
    evaluate_on_test(best_model, yaml_path)

    # Step 5: Download model terbaik
    print("\nDownload model dari Colab:")
    print("  from google.colab import files")
    print("  files.download('/content/runs/detect/student_engagement_v4/weights/best.pt')")
