from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

best_pt     = "/kaggle/working/runs/engagement_v7_classify/weights/best.pt"
DATA_DIR    = Path("/kaggle/working/crops_v7")
CLASSES     = ["Engaged", "NotEngaged"]
INFER_BATCH = 64

model    = YOLO(best_pt)
test_dir = DATA_DIR / "test"

y_true, y_pred = [], []
for cls_idx, cls_name in enumerate(CLASSES):
    cls_dir   = test_dir / cls_name
    img_files = list(cls_dir.glob("*.jpg"))
    print(f"Inferensi {cls_name}: {len(img_files)} gambar ...")
    for i in range(0, len(img_files), INFER_BATCH):
        batch = img_files[i : i + INFER_BATCH]
        for r in model.predict(batch, verbose=False, imgsz=224, stream=True):
            y_true.append(cls_idx)
            y_pred.append(int(r.probs.top1))

print("\n" + "=" * 55)
print("PER-CLASS REPORT V7 (Test Set)")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

cm = confusion_matrix(y_true, y_pred)
print("CONFUSION MATRIX (baris=aktual, kolom=prediksi):")
header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
print(header)
for i, row in enumerate(cm):
    print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14}" for v in row))

acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
print(f"\nOverall Top-1 Accuracy: {acc:.3f}")
