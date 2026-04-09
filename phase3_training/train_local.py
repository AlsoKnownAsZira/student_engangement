from ultralytics import YOLO
import os

def main():
    # 1. Path ke dataset yaml hasil split kita
    # Pastikan data.yaml berada di folder dataset_yolo_split
    dataset_yaml = os.path.abspath("../dataset_yolo_split/data.yaml")
    
    print(f"[INFO] Memulai training dengan dataset: {dataset_yaml}")
    
    # 2. Load model pre-trained bawaan Ultralytics (Bukan model buta Roboflow)
    # yolo11s.pt sudah punya otak untuk mengenali manusia (person) dengan sangat presisi
    print("[INFO] Memuat pre-trained model COCO yolo11s.pt...")
    model = YOLO("yolo11s.pt")
    
    # 3. Mulai proses Fine-Tuning
    # Model akan belajar menyesuaikan pemahamannya tentang "person" menjadi "Student Engagement" (Low, Medium, High)
    print("\n[INFO] >>> MEMULAI PROSES TRAINING LOKAL DENGAN AUGMENTASI <<<")
    results = model.train(
        data=dataset_yaml,
        epochs=70,             # Jumlah perulangan belajar
        imgsz=640,            # Ukuran gambar
        batch=16,             # Ukuran batch
        name="student_engagement_aug", # Ganti folder run name
        device=0,             # GPU Utama
        patience=15,          # Early stopping
        save=True,
        workers=4,            # Workers CPU
        
        # ========================================================
        # 🌟 HYPERPARAMETER AUGMENTASI ONLINE (ANTI OVERFITTING)
        # Model YOLO tidak menduplikasi file ke harddisk, tapi secara 
        # acak memanipulasi gambar "on-the-fly" di dalam RAM per putaran.
        # ========================================================
        hsv_s=0.7,         # Mengubah saturasi (Untuk kebal terhadap pencahayaan kelas berbeda)
        hsv_v=0.4,         # Mengubah kecerahan (value/brightness)
        degrees=10.0,      # Rotasi gambar acak -10 hingga 10 derajat
        translate=0.1,     # Geser gambar (panning) 10%
        scale=0.5,         # Zoom-in / Zoom-out acak 50%
        flipud=0.0,        # DILARANG flip vertikal (orang tidak mungkin terbalik)
        fliplr=0.5,        # 50% peluang dipantulkan layaknya cermin
        mosaic=1.0,        # (Default) 100% peluang menempelkan 4 gambar jadi 1 agar belajar konteks ramai
        mixup=0.15,        # 15% peluang gambar A dan B ditumpuk semi-transparan
        erasing=0.2        # 20% menutupi 1 kotak hitam acak (Melatih model kebal objek tertutup meja)
    )
    
    print("\n[INFO] Training Selesai! Model terbaik (best.pt) tersimpan di direktori 'runs/detect/student_engagement_aug/weights/'")

if __name__ == '__main__':
    main()
