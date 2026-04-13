from ultralytics import YOLO
import os


def main():
    import pathlib
    base_dir = pathlib.Path(__file__).parent.parent
    
    # Path ke model terbaik kita
    model_path = str(base_dir / "models" / "local_best.pt")
    
    # Path ke dataset yaml kita
    dataset_yaml = str(base_dir / "dataset_yolo_split" / "data.yaml")
    
    print(f"Loading Model: {model_path}")
    print(f"Loading Dataset: {dataset_yaml}")
    
    # Muat otak AI kita
    model = YOLO(model_path)
    
    print("\n🔥 --- MEMULAI EVALUASI (VALIDATION) PADA DATA TEST --- 🔥")
    print("Mengevaluasi kemampuan sesungguhnya pada sesi hari yang belum pernah model lihat (Unseen Split).")
    
    # ============================
    # Evaluasi STANDARD
    # ============================
    print("\n📊 [1/2] Evaluasi STANDARD (tanpa TTA):")
    metrics = model.val(
        data=dataset_yaml,
        split='test',  # Sangat penting: kita paksa tes di Unseen data, bukan di data Train/Val
        plots=True,    # Instruksi sakti! Akan otomatis menggambar Confusion Matrix, F1-Curve, dan PR-Curve
        save_json=True,
        name="skripsi_evaluation_test"
    )

    print(f"\n  mAP@50     : {metrics.box.map50:.3f}")
    print(f"  mAP@50-95  : {metrics.box.map:.3f}")
    
    # ============================
    # Evaluasi dengan TTA (Test-Time Augmentation)
    # Multi-scale testing: model dijalankan pada beberapa skala 
    # dan hasilnya digabung. Meningkatkan deteksi objek kecil ~2-5% mAP.
    # ============================
    print("\n📊 [2/2] Evaluasi dengan TTA (Test-Time Augmentation):")
    metrics_tta = model.val(
        data=dataset_yaml,
        split='test',
        augment=True,   # ✅ Multi-scale TTA
        plots=True,
        save_json=True,
        name="skripsi_evaluation_test_tta"
    )
    
    print(f"\n  mAP@50     : {metrics_tta.box.map50:.3f}")
    print(f"  mAP@50-95  : {metrics_tta.box.map:.3f}")
    
    # ============================
    # Perbandingan
    # ============================
    print("\n" + "=" * 50)
    print("📊 PERBANDINGAN HASIL")
    print("=" * 50)
    print(f"{'Metric':<20} {'Standard':>10} {'TTA':>10} {'Diff':>10}")
    print("-" * 52)
    
    diff_map50 = metrics_tta.box.map50 - metrics.box.map50
    diff_map = metrics_tta.box.map - metrics.box.map
    
    print(f"{'mAP@50':<20} {metrics.box.map50:>10.3f} {metrics_tta.box.map50:>10.3f} {diff_map50:>+10.3f}")
    print(f"{'mAP@50-95':<20} {metrics.box.map:>10.3f} {metrics_tta.box.map:>10.3f} {diff_map:>+10.3f}")
    print("=" * 50)

    print("\n✅ Evaluasi Selesai!")
    print(f"Confusion Matrix & Grafik Penilaian telah di-generate secara otomatis! 📊")
    print("Silakan buka folder:")
    print("  Standard : phase3_training/runs/detect/skripsi_evaluation_test/")
    print("  TTA      : phase3_training/runs/detect/skripsi_evaluation_test_tta/")
    

if __name__ == '__main__':
    main()
