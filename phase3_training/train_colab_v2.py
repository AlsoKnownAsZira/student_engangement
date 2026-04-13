"""
=============================================================================
 OPTIMIZED TRAINING SCRIPT v2 - Google Colab (Tesla T4)
=============================================================================
 Perubahan kunci dari v1:
   1. YOLO11m (20M params) menggantikan YOLO11s (9.4M params)
   2. Resolusi 960px (dari 640px) → objek kecil terlihat lebih jelas
   3. cls loss weight 1.0 (dari 0.5) → fokus klasifikasi engagement
   4. Patience 30 & epochs 150 → beri waktu model konvergen
   5. Cosine LR + warmup 5 epoch → stabilkan awal training
   6. copy_paste augmentation → variasi penempatan siswa
   7. dropout 0.1 + weight_decay 0.001 → anti overfitting
=============================================================================
 USAGE (di Google Colab):
   1. Upload dataset_yolo_split.zip ke Colab
   2. Jalankan cell-cell di bawah secara berurutan
=============================================================================
"""

from ultralytics import YOLO
import os
import shutil


def setup_dataset(zip_path="/content/dataset_yolo_split.zip", extract_to="/content"):
    """Ekstrak dataset dan update path di data.yaml"""
    import zipfile
    
    dataset_dir = os.path.join(extract_to, "dataset_yolo_split")
    
    if not os.path.exists(dataset_dir):
        print(f"[INFO] Mengekstrak dataset dari {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[INFO] Ekstraksi selesai!")
    else:
        print("[INFO] Dataset sudah terekstrak, skip.")
    
    # Update data.yaml path untuk Colab
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: train/images\n")
        f.write("val:   valid/images\n")
        f.write("test:  test/images\n")
        f.write("\n")
        f.write("nc: 3\n")
        f.write("names: ['High', 'Low', 'Medium']\n")
    
    print(f"[INFO] data.yaml updated dengan path: {dataset_dir}")
    return yaml_path


def count_dataset(dataset_dir="/content/dataset_yolo_split"):
    """Hitung distribusi kelas di dataset"""
    splits = ['train', 'valid', 'test']
    class_names = {0: 'High', 1: 'Low', 2: 'Medium'}
    
    print("\n" + "=" * 60)
    print("📊 DISTRIBUSI DATASET")
    print("=" * 60)
    
    for split in splits:
        label_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(label_dir):
            continue
        
        counts = {0: 0, 1: 0, 2: 0}
        n_images = 0
        
        for fname in os.listdir(label_dir):
            if fname.endswith('.txt'):
                n_images += 1
                with open(os.path.join(label_dir, fname)) as f:
                    for line in f:
                        cls = int(line.strip().split()[0])
                        if cls in counts:
                            counts[cls] += 1
        
        total = sum(counts.values())
        print(f"\n  [{split.upper()}] {n_images} images, {total} annotations")
        for cls_id, name in class_names.items():
            pct = counts[cls_id] / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {name:8s}: {counts[cls_id]:5d} ({pct:5.1f}%) {bar}")
    
    print("=" * 60)


def train_v2(dataset_yaml, model_size="yolo11m.pt"):
    """
    Training YOLO11m dengan semua optimisasi.
    
    Args:
        dataset_yaml: Path ke data.yaml
        model_size: Model pre-trained. Opsi:
            - "yolo11m.pt"  → Rekomendasi (20M params, balance)
            - "yolo11l.pt"  → Jika VRAM cukup (25M params, lebih akurat)
            - "yolo11s.pt"  → Fallback jika VRAM terbatas
    """
    print(f"\n[INFO] Loading pre-trained model: {model_size}")
    model = YOLO(model_size)
    
    print("\n" + "=" * 60)
    print("🚀 MEMULAI TRAINING v2 - ALL IMPROVEMENTS APPLIED")
    print("=" * 60)
    print(f"  Model       : {model_size}")
    print(f"  Resolution  : 960px")
    print(f"  Epochs      : 150 (patience=30)")
    print(f"  cls weight  : 1.0 (fokus klasifikasi)")
    print(f"  Augmentasi  : Mosaic + Mixup + CopyPaste + Erasing")
    print("=" * 60)
    
    results = model.train(
        data=dataset_yaml,
        
        # === CORE ===
        epochs=150,            # ⬆️ dari 70 → beri waktu konvergen penuh
        imgsz=960,             # ⬆️ dari 640 → kritis untuk objek kecil (CCTV)
        batch=8,               # ⬇️ dari 16 → kompensasi VRAM untuk 960px + model M
        device=0,              # GPU
        workers=4,             # Colab CPU workers
        
        # === EARLY STOPPING ===
        patience=30,           # ⬆️ dari 15 → jangan terlalu cepat menyerah
        
        # === LEARNING RATE ===
        lr0=0.001,             # Explicit (jangan auto)
        lrf=0.01,              # Final LR = lr0 * 0.01 = 0.00001
        cos_lr=True,           # ✅ Cosine annealing (lebih smooth dari linear decay)
        warmup_epochs=5.0,     # ⬆️ dari 3 → stabilkan gradient di awal
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # === LOSS WEIGHTS (KUNCI!) ===
        box=7.5,               # Keep default
        cls=1.0,               # ⬆️ dari 0.5 → LEBIH FOKUS ke klasifikasi engagement!
        dfl=1.5,               # Keep default
        
        # === AUGMENTASI ONLINE ===
        mosaic=1.0,            # 100% mosaic (4 gambar jadi 1)
        mixup=0.2,             # ⬆️ dari 0.15 → lebih banyak blending
        copy_paste=0.1,        # ✅ BARU! Copy-paste siswa ke posisi lain
        erasing=0.3,           # ⬆️ dari 0.2 → simulasi oklusi lebih agresif
        
        degrees=15.0,          # ⬆️ dari 10 → CCTV angle bervariasi
        translate=0.15,        # ⬆️ dari 0.1
        scale=0.5,             # Keep (zoom in/out 50%)
        shear=2.0,             # ✅ BARU! Sedikit distorsi perspektif
        
        fliplr=0.5,            # Flip horizontal 50%
        flipud=0.0,            # Disabled (orang tidak terbalik)
        
        hsv_h=0.02,            # ✅ BARU! Variasi warna (hue)
        hsv_s=0.7,             # Keep
        hsv_v=0.4,             # Keep
        
        close_mosaic=20,       # ⬆️ dari 10 → fine-tune tanpa mosaic lebih lama
        
        # === REGULARISASI (ANTI OVERFITTING) ===
        weight_decay=0.001,    # ⬆️ dari 0.0005
        dropout=0.1,           # ✅ BARU! Dropout di head
        
        # === SAVING ===
        name="student_engagement_v2",
        save=True,
        save_period=10,        # Simpan checkpoint setiap 10 epoch
        plots=True,            # Generate semua plot evaluasi
        
        # === DETERMINISTIC ===
        seed=42,
        deterministic=True,
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING SELESAI!")
    print("=" * 60)
    print(f"Model terbaik: runs/detect/student_engagement_v2/weights/best.pt")
    print(f"Model terakhir: runs/detect/student_engagement_v2/weights/last.pt")
    
    return results


def evaluate_on_test(model_path, dataset_yaml):
    """Evaluasi model pada test set (unseen data)"""
    print(f"\n[INFO] Evaluasi model: {model_path}")
    model = YOLO(model_path)
    
    # Standard evaluation
    print("\n📊 Evaluasi STANDARD (tanpa TTA):")
    metrics_standard = model.val(
        data=dataset_yaml,
        split='test',
        plots=True,
        save_json=True,
        name="eval_v2_standard"
    )
    
    # TTA evaluation (Test-Time Augmentation) 
    print("\n📊 Evaluasi dengan TTA (Test-Time Augmentation):")
    metrics_tta = model.val(
        data=dataset_yaml,
        split='test',
        augment=True,       # ✅ Multi-scale TTA
        plots=True,
        save_json=True,
        name="eval_v2_tta"
    )
    
    print("\n" + "=" * 60)
    print("📊 PERBANDINGAN HASIL")
    print("=" * 60)
    print(f"{'Metric':<20} {'Standard':>10} {'TTA':>10}")
    print("-" * 42)
    print(f"{'mAP@50':<20} {metrics_standard.box.map50:>10.3f} {metrics_tta.box.map50:>10.3f}")
    print(f"{'mAP@50-95':<20} {metrics_standard.box.map:>10.3f} {metrics_tta.box.map:>10.3f}")
    print("=" * 60)
    
    return metrics_standard, metrics_tta


# =============================================================================
# MAIN EXECUTION (jalankan di Colab)
# =============================================================================
if __name__ == "__main__":
    # Step 1: Setup dataset
    yaml_path = setup_dataset()
    
    # Step 2: Lihat distribusi dataset
    count_dataset()
    
    # Step 3: Training!
    results = train_v2(yaml_path, model_size="yolo11m.pt")
    
    # Step 4: Evaluasi di test set
    best_model = "/content/runs/detect/student_engagement_v2/weights/best.pt"
    evaluate_on_test(best_model, yaml_path)
    
    # Step 5: Download model terbaik
    print("\n💾 Download model dari Colab:")
    print("  from google.colab import files")
    print("  files.download('/content/runs/detect/student_engagement_v2/weights/best.pt')")
