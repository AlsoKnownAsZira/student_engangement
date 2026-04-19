"""
=============================================================================
 TRAINING SCRIPT v4 - Kaggle (Tesla T4 / P100)
=============================================================================
 Setup Kaggle sebelum run:
   1. Upload dataset_yolo_v4.zip sebagai Kaggle Dataset
      → Kaggle akan auto-extract, isi folder akan ada di /kaggle/input/<nama>/
   2. Di notebook Kaggle:
      - Settings > Accelerator: GPU T4 x2 atau P100
      - Attach dataset di atas
      - Upload script ini lalu run

 Training dari scratch dengan yolo11m.pt pretrained ImageNet.
 Output disimpan di /kaggle/working/ → download langsung dari panel Kaggle.
=============================================================================
"""

from ultralytics import YOLO
import os
import glob
import yaml


GDRIVE_FILE_ID = "GANTI_DENGAN_FILE_ID_GDRIVE"  # <-- isi file ID dari link Google Drive

def download_from_gdrive(file_id, output_path="/kaggle/working/dataset_yolo_v4.zip"):
    """Download dataset dari Google Drive pakai gdown"""
    if os.path.exists(output_path):
        print(f"[INFO] File sudah ada, skip download: {output_path}")
        return output_path

    print(f"[INFO] Downloading dari Google Drive (ID: {file_id})...")
    os.system(f"pip install gdown -q")
    import gdown
    gdown.download(id=file_id, output=output_path, quiet=False)
    print(f"[INFO] Download selesai: {output_path}")
    return output_path


def extract_dataset(zip_path="/kaggle/working/dataset_yolo_v4.zip",
                    extract_to="/kaggle/working"):
    """Ekstrak dataset zip"""
    import zipfile
    dataset_dir = os.path.join(extract_to, "dataset_yolo_split")

    if os.path.exists(dataset_dir):
        print(f"[INFO] Dataset sudah terekstrak: {dataset_dir}")
        return dataset_dir

    print(f"[INFO] Mengekstrak {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"[INFO] Ekstraksi selesai: {dataset_dir}")
    return dataset_dir



def setup_yaml(dataset_dir):
    """Tulis data_v4.yaml dengan path Kaggle yang benar"""
    yaml_path = "/kaggle/working/data_v4.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train_v4/images\n")
        f.write("val:   valid/images\n")
        f.write("test:  test/images\n")
        f.write("\n")
        f.write("nc: 3\n")
        f.write("names: ['High', 'Low', 'Medium']\n")
    print(f"[INFO] data_v4.yaml ditulis: {yaml_path}")
    print(f"[INFO] Dataset dir: {dataset_dir}")
    return yaml_path


def count_dataset(dataset_dir):
    """Hitung distribusi kelas"""
    splits = ["train_v4", "valid", "test"]
    class_names = {0: "High", 1: "Low", 2: "Medium"}

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
            if fname.endswith(".txt"):
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


def train_v4_kaggle(dataset_yaml, model_size="yolo11m.pt"):
    """Training dari scratch dengan yolo11m pretrained ImageNet"""
    print(f"\n[INFO] Loading pre-trained model: {model_size}")
    model = YOLO(model_size)

    print("\n" + "=" * 60)
    print("TRAINING v4 - KAGGLE")
    print("=" * 60)
    print(f"  Model       : {model_size}")
    print(f"  Resolution  : 960px")
    print(f"  Epochs      : 150")
    print(f"  Batch       : 6")
    print(f"  patience    : 70")
    print(f"  cls weight  : 1.5")
    print(f"  copy_paste  : 0.1")
    print(f"  label_smooth: 0.1")
    print(f"  close_mosaic: 30")
    print("=" * 60)

    results = model.train(
        data=dataset_yaml,

        # === CORE ===
        epochs=150,
        imgsz=960,
        batch=6,
        device=0,
        workers=2,          # Kaggle lebih stabil dengan workers lebih sedikit

        # === EARLY STOPPING ===
        patience=70,

        # === LEARNING RATE ===
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # === LOSS WEIGHTS ===
        box=7.5,
        cls=1.5,
        dfl=1.5,

        # === AUGMENTASI ONLINE ===
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
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

        close_mosaic=30,

        # === REGULARISASI ===
        weight_decay=0.001,
        dropout=0.1,

        # === SAVING ===
        project="/kaggle/working/runs",
        name="student_engagement_v4_kaggle",
        save=True,
        save_period=10,
        plots=True,

        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )

    return results


def evaluate_on_test(model_path, dataset_yaml):
    """Evaluasi model pada test set"""
    print(f"\n[INFO] Evaluasi: {model_path}")
    model = YOLO(model_path)

    print("\nEvaluasi STANDARD:")
    metrics = model.val(
        data=dataset_yaml,
        split="test",
        plots=True,
        save_json=True,
        project="/kaggle/working/runs",
        name="eval_v4_standard",
    )

    print("\nEvaluasi TTA:")
    metrics_tta = model.val(
        data=dataset_yaml,
        split="test",
        augment=True,
        plots=True,
        save_json=True,
        project="/kaggle/working/runs",
        name="eval_v4_tta",
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI V4")
    print("=" * 60)
    print(f"{'Metric':<20} {'Standard':>10} {'TTA':>10}")
    print("-" * 42)
    print(f"{'mAP@50':<20} {metrics.box.map50:>10.3f} {metrics_tta.box.map50:>10.3f}")
    print(f"{'mAP@50-95':<20} {metrics.box.map:>10.3f} {metrics_tta.box.map:>10.3f}")

    print("\nPer-class mAP@50:")
    class_names = ["High", "Low", "Medium"]
    for i, name in enumerate(class_names):
        try:
            print(f"  {name:8s}: {metrics.box.ap50[i]:.3f}")
        except Exception:
            pass
    print("=" * 60)

    return metrics, metrics_tta


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Step 1: Download & ekstrak dari Google Drive
    zip_path = download_from_gdrive(GDRIVE_FILE_ID)
    dataset_dir = extract_dataset(zip_path)
    print(f"[INFO] Dataset  : {dataset_dir}")

    # Step 2: Setup YAML
    yaml_path = setup_yaml(dataset_dir)

    # Step 3: Distribusi dataset
    count_dataset(dataset_dir)

    # Step 4: Training
    results = train_v4_kaggle(yaml_path)

    # Step 5: Evaluasi
    best_model = "/kaggle/working/runs/student_engagement_v4_kaggle/weights/best.pt"
    evaluate_on_test(best_model, yaml_path)

    print("\n" + "=" * 60)
    print("SELESAI! Download model dari:")
    print("  /kaggle/working/runs/student_engagement_v4_kaggle/weights/best.pt")
    print("=" * 60)
