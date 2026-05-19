"""
Evaluasi model deteksi V5 (best_v5.pt) pada test split dataset_yolo_v5.
Output: mAP@50, mAP@50-95, Precision, Recall per kelas (Engaged & NotEngaged).
Confusion matrix dan kurva PR disimpan otomatis oleh YOLO ke folder runs/.
"""

import pathlib
import tempfile
import yaml
from ultralytics import YOLO

BASE_DIR = pathlib.Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_v5.pt"
DATASET_DIR = BASE_DIR / "unused" / "dataset_yolo_v5"


def make_fixed_yaml(dataset_dir: pathlib.Path) -> str:
    """Buat YAML sementara dengan path absolut yang benar."""
    cfg = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc": 2,
        "names": ["Engaged", "NotEngaged"],
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg, tmp, allow_unicode=True)
    tmp.close()
    return tmp.name


def main():
    print(f"Model : {MODEL_PATH}")
    print(f"Dataset: {DATASET_DIR}")

    yaml_path = make_fixed_yaml(DATASET_DIR)
    model = YOLO(str(MODEL_PATH))

    print("\n[1/2] Evaluasi STANDARD (tanpa TTA)")
    m = model.val(
        data=yaml_path,
        split="test",
        plots=True,
        save_json=False,
        name="eval_v5_test",
    )

    names = ["Engaged", "NotEngaged"]

    print("\n" + "=" * 55)
    print(f"{'Metrik':<25} {'Nilai':>10}")
    print("-" * 55)
    print(f"{'mAP@50 (overall)':<25} {m.box.map50:>10.4f}")
    print(f"{'mAP@50-95 (overall)':<25} {m.box.map:>10.4f}")

    # Per-class mAP
    if hasattr(m.box, 'maps') and m.box.maps is not None:
        for i, name in enumerate(names):
            if i < len(m.box.maps):
                print(f"{'mAP@50 ' + name:<25} {m.box.maps[i]:>10.4f}")

    # Per-class Precision & Recall
    # m.box.p  → precision per class (numpy array)
    # m.box.r  → recall per class (numpy array)
    print()
    if hasattr(m.box, 'p') and m.box.p is not None:
        p_vals = m.box.p  # shape: (num_classes,) at IoU=0.5
        r_vals = m.box.r
        for i, name in enumerate(names):
            if i < len(p_vals):
                print(f"{'Precision ' + name:<25} {float(p_vals[i]):>10.4f}")
                print(f"{'Recall    ' + name:<25} {float(r_vals[i]):>10.4f}")
    else:
        # Fallback — cetak raw metrics object
        print("Precision/Recall per kelas tidak tersedia langsung, cek folder runs/.")
        print(f"Precision mean : {m.box.mp:.4f}")
        print(f"Recall mean    : {m.box.mr:.4f}")

    print("=" * 55)
    print(f"\nConfusion matrix & grafik disimpan di:")
    print(f"  phase3_training/runs/detect/eval_v5_test/")


if __name__ == "__main__":
    main()
