"""
=============================================================================
 TRAINING SCRIPT v5 - Kaggle (2-class: Engaged, NotEngaged)
=============================================================================
 Key differences vs v4:
   - 2 classes (High+Medium digabung jadi Engaged; Low -> NotEngaged)
   - Base dataset (TIDAK pakai augmentasi offline v4)
   - Focal loss (fl_gamma=1.5) untuk handle imbalance ringan ~65:35
   - cls weight kembali ke 1.0 (default) — untuk 2-class sudah cukup
   - label_smoothing=0.05 — lebih rendah karena batas kelas lebih tegas
   - imgsz=960, model yolo11m — konsisten dengan v2-v4 (proven plateau tester)

 Target: mAP@50 70-78% (vs plateau 54-59% di 3-class)

 Setup Kaggle:
   1. Upload folder `dataset_yolo_v5/` sebagai Kaggle Dataset (zip dulu lokal)
   2. Notebook settings: GPU T4 x2 atau P100, attach dataset
   3. Run script ini
=============================================================================
"""

from ultralytics import YOLO
import os
import zipfile


GDRIVE_FILE_ID = "GANTI_DENGAN_FILE_ID_GDRIVE"  # opsional, kalau upload via gdrive


def download_from_gdrive(file_id, output_path="/kaggle/working/dataset_yolo_v5.zip"):
    if os.path.exists(output_path):
        print(f"[INFO] File sudah ada: {output_path}")
        return output_path
    os.system("pip install gdown -q")
    import gdown
    gdown.download(id=file_id, output=output_path, quiet=False)
    return output_path


def extract_dataset(zip_path, extract_to="/kaggle/working"):
    dataset_dir = os.path.join(extract_to, "dataset_yolo_v5")
    if os.path.exists(dataset_dir):
        print(f"[INFO] Sudah terekstrak: {dataset_dir}")
        return dataset_dir
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    return dataset_dir


def setup_yaml(dataset_dir):
    yaml_path = "/kaggle/working/data_v5.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train/images\n")
        f.write("val:   valid/images\n")
        f.write("test:  test/images\n\n")
        f.write("nc: 2\n")
        f.write("names: ['Engaged', 'NotEngaged']\n")
    print(f"[INFO] {yaml_path}")
    return yaml_path


def count_dataset(dataset_dir):
    splits = ["train", "valid", "test"]
    names = {0: "Engaged", 1: "NotEngaged"}
    print("\n" + "=" * 60)
    print("DISTRIBUSI DATASET V5 (2-class)")
    print("=" * 60)
    for split in splits:
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(lbl_dir):
            print(f"  [{split}] tidak ditemukan")
            continue
        counts = {0: 0, 1: 0}
        n_images = 0
        for fname in os.listdir(lbl_dir):
            if fname.endswith(".txt"):
                n_images += 1
                with open(os.path.join(lbl_dir, fname)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            if cls in counts:
                                counts[cls] += 1
        total = sum(counts.values())
        print(f"\n  [{split.upper()}] {n_images} images, {total} annotations")
        for cid, name in names.items():
            pct = counts[cid] / total * 100 if total else 0
            bar = "#" * int(pct / 2)
            print(f"    {name:11s}: {counts[cid]:5d} ({pct:5.1f}%) {bar}")
    print("=" * 60)


def train_v5_kaggle(dataset_yaml, model_size="yolo11m.pt"):
    print(f"\n[INFO] Loading {model_size}")
    model = YOLO(model_size)

    print("\n" + "=" * 60)
    print("TRAINING v5 - KAGGLE (2-class)")
    print("=" * 60)
    print(f"  Classes     : Engaged, NotEngaged")
    print(f"  Model       : {model_size}")
    print(f"  Resolution  : 960px")
    print(f"  Epochs      : 120 (patience 50)")
    print(f"  Batch       : 8")
    print(f"  cls weight  : 1.2")
    print(f"  label_smooth: 0.05")
    print(f"  copy_paste  : 0.2 (anti-imbalance)")
    print("=" * 60)

    results = model.train(
        data=dataset_yaml,

        # === CORE ===
        epochs=120,
        imgsz=960,
        batch=8,
        device=0,
        workers=2,
        cache="disk",

        # === EARLY STOPPING ===
        patience=50,

        # === LEARNING RATE ===
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # === LOSS ===
        # Note: fl_gamma dihapus di Ultralytics versi baru. Untuk imbalance
        # mild 2:1 ini, andalkan copy_paste=0.2 + cls bump kecil.
        box=7.5,
        cls=1.2,         # sedikit naik dari default 1.0 untuk kompensasi tanpa focal
        dfl=1.5,
        label_smoothing=0.05,

        # === AUGMENTASI ONLINE ===
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,  # naik dari 0.1 — bantu minority class via instance copy
        erasing=0.3,

        degrees=10.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,

        fliplr=0.5,
        flipud=0.0,

        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,

        close_mosaic=20,

        # === REGULARISASI ===
        weight_decay=0.001,
        dropout=0.1,

        # === SAVING ===
        project="/kaggle/working/runs",
        name="student_engagement_v5_kaggle",
        save=True,
        save_period=10,
        plots=True,

        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )

    return results


def evaluate_on_test(model_path, dataset_yaml):
    print(f"\n[INFO] Evaluasi: {model_path}")
    model = YOLO(model_path)

    print("\nSTANDARD:")
    m = model.val(
        data=dataset_yaml, split="test", plots=True, save_json=True,
        project="/kaggle/working/runs", name="eval_v5_standard",
    )

    print("\nTTA:")
    m_tta = model.val(
        data=dataset_yaml, split="test", augment=True, plots=True, save_json=True,
        project="/kaggle/working/runs", name="eval_v5_tta",
    )

    print("\n" + "=" * 60)
    print("HASIL V5")
    print("=" * 60)
    print(f"{'Metric':<20} {'Standard':>10} {'TTA':>10}")
    print("-" * 42)
    print(f"{'mAP@50':<20} {m.box.map50:>10.3f} {m_tta.box.map50:>10.3f}")
    print(f"{'mAP@50-95':<20} {m.box.map:>10.3f} {m_tta.box.map:>10.3f}")

    print("\nPer-class mAP@50:")
    for i, name in enumerate(["Engaged", "NotEngaged"]):
        try:
            print(f"  {name:11s}: {m.box.ap50[i]:.3f}")
        except Exception:
            pass
    print("=" * 60)
    return m, m_tta


if __name__ == "__main__":
    zip_path = download_from_gdrive(GDRIVE_FILE_ID)
    dataset_dir = extract_dataset(zip_path)
    yaml_path = setup_yaml(dataset_dir)
    count_dataset(dataset_dir)
    results = train_v5_kaggle(yaml_path)
    best = "/kaggle/working/runs/student_engagement_v5_kaggle/weights/best.pt"
    evaluate_on_test(best, yaml_path)
    print("\nSELESAI. Best weights:")
    print(f"  {best}")
