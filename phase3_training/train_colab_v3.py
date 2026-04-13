"""
=============================================================================
 OPTIMIZED TRAINING SCRIPT v3 - Google Colab (Tesla T4)
=============================================================================
 Fokus Utama V3: Resolusi Ekstrem & Penalti Klasifikasi (Menaklukkan M@P 57%)

 Mengapa V3 berbeda dari V2?
   1. Resolusi naik ke 1280px (maksimal untuk Tesla T4 dengan batch 4). 
      Jauh lebih baik dari 960px untuk melihat wajah siswa di pojok. Efek resolusi 
      tinggi ini sejalan dengan tujuan SAHI untuk mendeteksi objek kecil.
   2. cls loss weight 1.5 (dari 1.0) -> Memaksa model sangat fokus pada perbedaan High/Medium/Low, memberi efek jera saat model kebingungan.
   3. Copy-Paste 0.2 & MixUp 0.2 -> Memperkaya posisi kelas Medium yang jumlah datasetnya paling kecil (25%).
   4. Mempertahankan augmentasi warna (HSV) & rotasi agar kebal perubahan cahaya.
=============================================================================
 USAGE:
   1. Pastikan Anda mengekspor dataset terbaru (1117 images) dari Roboflow.
   2. Jalankan split_by_session.py di laptop Anda.
   3. Zip foldernya menjadi dataset_yolo_split.zip.
   4. Upload ke Google Colab, lalu jalankan script ini sel cell demi sel.
=============================================================================
"""

from ultralytics import YOLO
import os
import shutil


def setup_dataset(zip_path="/content/dataset_yolo_split.zip", extract_to="/content"):
    """Ekstrak dataset dan update path di data.yaml"""
    import zipfile
    
    dataset_dir = os.path.join(extract_to, "dataset_yolo_split")
    
    if os.path.exists(dataset_dir):
        print("[INFO] Dataset lama ditemukan, menghapus untuk extraction yang baru...")
        shutil.rmtree(dataset_dir)
        
    print(f"[INFO] Mengekstrak dataset dari {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[INFO] Ekstraksi selesai!")
    
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
    """Hitung distribusi kelas di dataset V3"""
    splits = ['train', 'valid', 'test']
    class_names = {0: 'High', 1: 'Low', 2: 'Medium'}
    
    print("\n" + "=" * 60)
    print("📊 DISTRIBUSI DATASET V3 TERBARU")
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
        if total > 0:
            for cls_id, name in class_names.items():
                pct = counts[cls_id] / total * 100
                bar = "█" * int(pct / 2)
                print(f"    {name:8s}: {counts[cls_id]:5d} ({pct:5.1f}%) {bar}")
    
    print("=" * 60)


def train_v3(dataset_yaml, model_size="yolo11m.pt"):
    """
    Training YOLO11m dengan setup V3: Resolusi 1280px
    """
    print(f"\n[INFO] Loading pre-trained model: {model_size}")
    model = YOLO(model_size)
    
    print("\n" + "=" * 60)
    print("🚀 MEMULAI TRAINING v3 - HIGH RESOLUTION & HIGH CLS PENALTY")
    print("=" * 60)
    
    results = model.train(
        data=dataset_yaml,
        
        # === KUNCI UTAMA V3: RESOLUSI EKSTREM ===
        imgsz=1280,            # ⬆️ V2=960. V3 mentok 1280px agar pendekatan deteksi sekelas SAHI dicapai secara native!
        batch=4,               # ⬇️ V2=8. Diturunkan agar VRAM Tesla T4 mumpuni untuk gambar ukuran 1280px.
        
        # === CORE ===
        epochs=150,            
        device=0,              
        workers=4,             
        
        # === EARLY STOPPING ===
        patience=40,           # ⬆️ V2=30. Diperlama karena model perlu lebih adaptasi dengan gambar besar.
        
        # === LEARNING RATE ===
        lr0=0.001,             
        lrf=0.01,              
        cos_lr=True,           
        warmup_epochs=4.0,     
        
        # === LOSS WEIGHTS (HUKUMAN SAAT MODEL SALAH KLASIFIKASI) ===
        box=7.5,               
        cls=1.5,               # ⬆️ V2=1.0. Sekarang 1.5! Kalau deteksi benar tapi klasifikasi salah, loss akan membengkak, memaksa model lebih teliti.
        dfl=1.5,               
        
        # === AUGMENTASI ONLINE ===
        mosaic=1.0,            
        mixup=0.2,             
        copy_paste=0.2,        # ⬆️ v2=0.1. Menempelkan banyak kelas minoritas di kanvas
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
        
        close_mosaic=20,       
        
        # === REGULARISASI ===
        weight_decay=0.001,    
        dropout=0.1,           
        
        # === SAVING ===
        name="student_engagement_v3",
        save=True,
        plots=True,
    )
    return results


def evaluate_on_test(model_path, dataset_yaml):
    """Evaluasi model pada test set (unseen data) menggunakan TTA yang native disupport"""
    print(f"\n[INFO] Evaluasi model V3: {model_path}")
    model = YOLO(model_path)
    
    # Standard evaluation pada resolusi asli training (1280px)
    print("\n📊 Evaluasi V3 STANDARD:")
    metrics_standard = model.val(
        data=dataset_yaml,
        split='test',
        imgsz=1280,
        plots=True,
        name="eval_v3_standard"
    )
    
    # TTA evaluation (Test-Time Augmentation) -> Sebagai pengganti instan SAHI di sisi Inference YOLO
    print("\n📊 Evaluasi V3 TTA (Test-Time Augmentation):")
    metrics_tta = model.val(
        data=dataset_yaml,
        split='test',
        imgsz=1280,
        augment=True,       # Menguji di berbagai skala secara otomatis
        plots=True,
        name="eval_v3_tta"
    )
    
    print("\n" + "=" * 60)
    print("📊 PERBANDINGAN HASIL V3")
    print("=" * 60)
    print(f"{'Metric':<20} {'Standard':>10} {'TTA':>10}")
    print("-" * 42)
    print(f"{'mAP@50':<20} {metrics_standard.box.map50:>10.3f} {metrics_tta.box.map50:>10.3f}")
    print(f"{'mAP@50-95':<20} {metrics_standard.box.map:>10.3f} {metrics_tta.box.map:>10.3f}")
    print("=" * 60)
    

if __name__ == "__main__":
    # 1. Setup dataset (Karena sudah Anda lakukan di cell sebelumnya via Google Drive,
    # kita lewati proses ekstraksi setup_dataset() agar tidak error).
    yaml_path = '/content/dataset_yolo_split/data.yaml'
    
    # 2. Cek integritas data
    count_dataset()
    
    # 3. Training V3
    train_v3(yaml_path, model_size="yolo11m.pt")
    
    # 4. Evaluasi (TTA vs Standard)
    best_model = "/content/runs/detect/student_engagement_v3/weights/best.pt"
    evaluate_on_test(best_model, yaml_path)
    
    print("\n💾 Download model V3 dari Colab:")
    print("  from google.colab import files")
    print("  files.download('/content/runs/detect/student_engagement_v3/weights/best.pt')")
