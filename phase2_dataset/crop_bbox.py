"""
Crop setiap bounding box dari dataset_yolo_split ke folder per kelas.

Input : dataset_yolo_split/{train,valid,test}/{images,labels}
Output: phase2_dataset/crops_v5/{train,valid,test}/{High,Low,Medium}/

Format label YOLO: class cx cy w h  (semua 0-1 relatif terhadap gambar)
Padding: 10% dari dimensi bbox di setiap sisi agar ada sedikit konteks.
"""

import os
from pathlib import Path
from PIL import Image
import collections

ROOT      = Path(__file__).resolve().parent.parent
SRC       = ROOT / "dataset_yolo_split"
DST       = ROOT / "phase2_dataset" / "crops_v6"
SPLITS    = ["train", "valid", "test"]
CLASS_MAP = {0: "High", 1: "Low", 2: "Medium"}
PADDING   = 0.10  # 10% padding relatif terhadap bbox w/h


def crop_split(split: str) -> dict:
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    counts  = collections.Counter()
    skipped = 0

    label_files = list(lbl_dir.glob("*.txt"))
    total = len(label_files)

    for i, lbl_path in enumerate(label_files, 1):
        if i % 100 == 0 or i == total:
            print(f"  [{split}] {i}/{total} files diproses ...", end="\r")

        # Cari file gambar yang cocok
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = img_dir / (lbl_path.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        lines = lbl_path.read_text().splitlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id not in CLASS_MAP:
                continue

            cx, cy, bw, bh = map(float, parts[1:5])

            # Konversi ke piksel
            cx_px = cx * W
            cy_px = cy * H
            bw_px = bw * W
            bh_px = bh * H

            # Tambah padding
            pad_x = PADDING * bw_px
            pad_y = PADDING * bh_px

            x1 = max(0, int(cx_px - bw_px / 2 - pad_x))
            y1 = max(0, int(cy_px - bh_px / 2 - pad_y))
            x2 = min(W, int(cx_px + bw_px / 2 + pad_x))
            y2 = min(H, int(cy_px + bh_px / 2 + pad_y))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            class_name = CLASS_MAP[cls_id]
            out_dir = DST / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{lbl_path.stem}__b{idx}.jpg"
            crop.save(out_dir / out_name, "JPEG", quality=95)
            counts[class_name] += 1

    print()  # newline setelah \r
    if skipped:
        print(f"  [WARN] {skipped} file label tidak ditemukan pasangan gambarnya")
    return counts


def main():
    print(f"Source : {SRC}")
    print(f"Output : {DST}\n")

    grand_total = collections.Counter()
    for split in SPLITS:
        print(f"[{split.upper()}]")
        counts = crop_split(split)
        total = sum(counts.values())
        for cls in ["High", "Low", "Medium"]:
            pct = 100 * counts[cls] / total if total else 0
            print(f"  {cls:8s}: {counts[cls]:5d} ({pct:.1f}%)")
        print(f"  Total   : {total}\n")
        grand_total += counts

    print("=== GRAND TOTAL ===")
    total_all = sum(grand_total.values())
    for cls in ["High", "Low", "Medium"]:
        pct = 100 * grand_total[cls] / total_all if total_all else 0
        print(f"  {cls:8s}: {grand_total[cls]:5d} ({pct:.1f}%)")
    print(f"  Total   : {total_all}")
    print(f"\nSelesai. Crops tersimpan di: {DST}")


if __name__ == "__main__":
    main()
