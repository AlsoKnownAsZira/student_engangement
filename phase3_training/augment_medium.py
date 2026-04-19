"""
augment_medium.py — Targeted Copy-Paste Augmentation untuk kelas Medium

Tujuan:
    Mengatasi class imbalance dengan mengaugmentasi HANYA bounding box
    berlabel Medium, lalu paste kembali ke frame yang ada.
    Hasilnya: dataset baru dengan Medium yang lebih seimbang.

Cara kerja:
    1. Scan semua train/labels/*.txt → kumpulkan semua bbox Medium
    2. Crop region siswa Medium dari train/images/
    3. Augmentasi crop (flip, brightness, contrast, rotate ringan)
    4. Paste crop yang sudah diaugmentasi ke frame donor (frame lain)
    5. Tulis image + label baru ke output_dir

Hasil output:
    Folder `dataset_yolo_split/train_v4/` berisi:
    - Semua frame original (copied)
    - Frame-frame baru hasil augmentasi

Cara pakai:
    python phase3_training/augment_medium.py
    python phase3_training/augment_medium.py --target 5000 --output dataset_yolo_split/train_v4

Dependencies:
    pip install albumentations opencv-python numpy tqdm
"""

import cv2
import numpy as np
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[WARNING] albumentations tidak terinstall. Pakai augmentasi manual.")
    print("          Install: pip install albumentations")


# ── Konfigurasi ──────────────────────────────────────────────────────────────

DATASET_ROOT = Path("dataset_yolo_split")
TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"

CLASS_NAMES  = {0: "High", 1: "Low", 2: "Medium"}

# Kelas yang akan diaugmentasi (bisa lebih dari satu)
# Key = class_id, Value = jumlah instance target
AUGMENT_TARGETS = {
    1: 5000,   # Low  : 3872 → 5000 (perlu +1128)
    2: 5000,   # Medium: 3366 → 5000 (perlu +1634)
}

# Padding di sekitar bbox saat crop (% dari ukuran bbox)
CROP_PADDING = 0.05

# Seed untuk reproducibility
RANDOM_SEED = 42


# ── Augmentation Pipeline ─────────────────────────────────────────────────────

def build_augmentor():
    """Augmentasi yang aman untuk crop siswa kelas Medium."""
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Rotate(limit=8, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.RandomScale(scale_limit=0.1, p=0.3),
        ])
    return None


def augment_crop_manual(crop: np.ndarray) -> np.ndarray:
    """Fallback augmentasi tanpa albumentations."""
    # Flip horizontal
    if random.random() > 0.5:
        crop = cv2.flip(crop, 1)
    # Brightness
    factor = random.uniform(0.75, 1.25)
    crop = np.clip(crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return crop


def augment_crop(crop: np.ndarray, augmentor) -> np.ndarray:
    if augmentor is not None:
        result = augmentor(image=crop)
        return result["image"]
    return augment_crop_manual(crop)


# ── Helpers ───────────────────────────────────────────────────────────────────

def yolo_to_pixel(box_yolo, img_w, img_h):
    """Convert YOLO (cx, cy, w, h) normalized ke pixel (x1, y1, x2, y2)."""
    cx, cy, bw, bh = box_yolo
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return (
        max(0, x1), max(0, y1),
        min(img_w, x2), min(img_h, y2)
    )


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel (x1, y1, x2, y2) ke YOLO normalized."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def load_labels(label_path: Path):
    """Baca YOLO label file, return list of (class_id, cx, cy, w, h)."""
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().split("\n")
    annotations = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:5]))
        annotations.append((cls, *coords))
    return annotations


def save_labels(label_path: Path, annotations):
    """Tulis list of (class_id, cx, cy, w, h) ke YOLO label file."""
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for cls, cx, cy, w, h in annotations]
    label_path.write_text("\n".join(lines))


# ── Core Logic ────────────────────────────────────────────────────────────────

def collect_crops_by_class(augmentor, n_augments_per_crop: int = 3):
    """
    Kumpulkan crop dari semua kelas yang ada di AUGMENT_TARGETS.

    Returns:
        dict: {class_id: [(crop_image, bw_ratio, bh_ratio), ...]}
    """
    print("\n[1/3] Mengumpulkan crops dari train set...")
    crops = {cls_id: [] for cls_id in AUGMENT_TARGETS}

    image_files = sorted(TRAIN_IMAGES.glob("*.jpg")) + \
                  sorted(TRAIN_IMAGES.glob("*.png"))

    for img_path in tqdm(image_files, desc="Scanning"):
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        annotations = load_labels(label_path)

        relevant = [a for a in annotations if a[0] in AUGMENT_TARGETS]
        if not relevant:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for ann in relevant:
            cls_id, cx, cy, bw, bh = ann
            x1, y1, x2, y2 = yolo_to_pixel((cx, cy, bw, bh), w, h)

            pad_x = int((x2 - x1) * CROP_PADDING)
            pad_y = int((y2 - y1) * CROP_PADDING)
            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)

            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            crops[cls_id].append((crop.copy(), bw, bh))
            for _ in range(n_augments_per_crop):
                crops[cls_id].append((augment_crop(crop, augmentor), bw, bh))

    for cls_id, cls_crops in crops.items():
        print(f"    {CLASS_NAMES[cls_id]:8s}: {len(cls_crops)} crops terkumpul")
    return crops


def compute_iou(box_a, box_b):
    """
    Hitung IoU antara dua bbox dalam format pixel (x1, y1, x2, y2).
    Digunakan untuk cek apakah posisi paste overlap dengan bbox yang sudah ada.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def find_empty_position(existing_anns, img_w, img_h,
                         paste_w, paste_h,
                         max_iou=0.1, max_attempts=30):
    """
    Cari posisi paste yang tidak overlap dengan bbox yang sudah ada di frame.

    Args:
        existing_anns : list of (cls_id, cx, cy, bw, bh) YOLO format
        img_w, img_h  : ukuran frame
        paste_w, paste_h : ukuran crop yang akan di-paste
        max_iou       : toleransi overlap maksimal (0.1 = 10%)
        max_attempts  : berapa kali coba sebelum menyerah

    Returns:
        (px, py) posisi top-left yang aman, atau None jika tidak ditemukan
    """
    # Konversi semua existing bbox ke pixel untuk perbandingan
    existing_pixels = []
    for ann in existing_anns:
        _, cx, cy, bw, bh = ann
        x1, y1, x2, y2 = yolo_to_pixel((cx, cy, bw, bh), img_w, img_h)
        existing_pixels.append((x1, y1, x2, y2))

    max_x = img_w - paste_w
    max_y = img_h - paste_h
    if max_x <= 0 or max_y <= 0:
        return None

    for _ in range(max_attempts):
        px = random.randint(0, max_x)
        py = random.randint(0, max_y)
        candidate = (px, py, px + paste_w, py + paste_h)

        # Cek overlap dengan semua bbox yang ada
        overlapping = any(
            compute_iou(candidate, existing) > max_iou
            for existing in existing_pixels
        )

        if not overlapping:
            return px, py

    return None  # Tidak ada posisi aman ditemukan di frame ini


def generate_augmented_frames(crops_by_class, output_images: Path, output_labels: Path):
    """
    Paste crops ke frame donor di posisi KOSONG (tidak overlap siswa lain).
    Tiap frame baru bisa berisi paste dari beberapa kelas sekaligus.
    """
    # Hitung berapa instance yang perlu di-generate per kelas
    existing_counts = {}
    for cls_id in AUGMENT_TARGETS:
        count = sum(
            1 for f in TRAIN_LABELS.glob("*.txt")
            for line in f.read_text().strip().split("\n")
            if line.strip() and int(line.split()[0]) == cls_id
        )
        existing_counts[cls_id] = count

    needed = {
        cls_id: max(0, AUGMENT_TARGETS[cls_id] - existing_counts[cls_id])
        for cls_id in AUGMENT_TARGETS
    }

    print(f"\n[2/3] Membuat frame baru (paste di area kosong)...")
    for cls_id, n in needed.items():
        print(f"    {CLASS_NAMES[cls_id]:8s}: perlu +{n} instance")

    total_needed = sum(needed.values())
    if total_needed == 0:
        print("    Semua kelas sudah mencapai target.")
        return 0

    image_files = sorted(TRAIN_IMAGES.glob("*.jpg")) + \
                  sorted(TRAIN_IMAGES.glob("*.png"))

    generated = {cls_id: 0 for cls_id in AUGMENT_TARGETS}
    frame_count = 0
    skipped_no_space = 0
    pbar = tqdm(total=total_needed, desc="Generating")

    while any(generated[c] < needed[c] for c in needed):
        donor_path = random.choice(image_files)
        label_path = TRAIN_LABELS / (donor_path.stem + ".txt")

        donor = cv2.imread(str(donor_path))
        if donor is None:
            continue
        img_h, img_w = donor.shape[:2]

        existing_anns = load_labels(label_path)
        # Salin anotasi yang ada — ini yang kita jaga agar tidak tertutup
        new_anns = list(existing_anns)
        new_img = donor.copy()
        pasted_total = 0

        for cls_id in AUGMENT_TARGETS:
            if generated[cls_id] >= needed[cls_id]:
                continue
            if not crops_by_class[cls_id]:
                continue

            n_paste = random.randint(1, 2)
            for _ in range(n_paste):
                if generated[cls_id] >= needed[cls_id]:
                    break

                crop, orig_bw, orig_bh = random.choice(crops_by_class[cls_id])

                target_w = int(orig_bw * img_w)
                target_h = int(orig_bh * img_h)
                if target_w < 10 or target_h < 10:
                    continue

                # Cari posisi yang tidak overlap dengan siswa yang sudah ada
                pos = find_empty_position(
                    new_anns, img_w, img_h,
                    target_w, target_h,
                    max_iou=0.1,
                    max_attempts=30
                )

                if pos is None:
                    # Frame ini penuh, tidak ada ruang kosong
                    skipped_no_space += 1
                    continue

                px, py = pos
                crop_resized = cv2.resize(crop, (target_w, target_h))
                new_img[py:py+target_h, px:px+target_w] = crop_resized

                cx, cy, bw, bh = pixel_to_yolo(
                    px, py, px+target_w, py+target_h, img_w, img_h
                )
                new_anns.append((cls_id, cx, cy, bw, bh))
                generated[cls_id] += 1
                pasted_total += 1
                pbar.update(1)

        if pasted_total == 0:
            continue

        out_name = f"aug_bal_{frame_count:05d}"
        cv2.imwrite(str(output_images / f"{out_name}.jpg"), new_img)
        save_labels(output_labels / f"{out_name}.txt", new_anns)
        frame_count += 1

    pbar.close()
    print(f"    Frame baru dibuat   : {frame_count}")
    print(f"    Paste gagal (penuh) : {skipped_no_space}")
    for cls_id, n in generated.items():
        print(f"    {CLASS_NAMES[cls_id]:8s} baru   : {n} instance")
    return frame_count


def copy_original_dataset(output_images: Path, output_labels: Path):
    """Copy semua frame original ke output folder."""
    print("\n[3/3] Menyalin dataset original...")
    image_files = sorted(TRAIN_IMAGES.glob("*.jpg")) + \
                  sorted(TRAIN_IMAGES.glob("*.png"))

    for img_path in tqdm(image_files, desc="Copying"):
        shutil.copy2(img_path, output_images / img_path.name)
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, output_labels / label_path.name)


def print_class_distribution(label_dir: Path, label: str):
    counts = {0: 0, 1: 0, 2: 0}
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().strip().split("\n"):
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if cls in counts:
                counts[cls] += 1
    total = sum(counts.values())
    print(f"\n  [{label}]")
    for cls_id, name in CLASS_NAMES.items():
        pct = counts[cls_id] / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {name:8s} ({cls_id}): {counts[cls_id]:5d} ({pct:.1f}%) {bar}")
    print(f"    Total         : {total}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Balanced class augmentation untuk dataset YOLO V4"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DATASET_ROOT / "train_v4"),
        help="Folder output train set V4 (default: dataset_yolo_split/train_v4)"
    )
    parser.add_argument(
        "--augments-per-crop", type=int, default=3,
        help="Berapa kali tiap crop diaugmentasi (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir    = Path(args.output)
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Balanced Class Augmentation untuk V4 Training")
    print("=" * 60)
    print(f"  Dataset  : {DATASET_ROOT}")
    print(f"  Output   : {output_dir}")
    print(f"  Target   : {AUGMENT_TARGETS}")
    print(f"  Aug/crop : {args.augments_per_crop}x per crop")

    # Distribusi sebelum
    print_class_distribution(TRAIN_LABELS, "SEBELUM (original train)")

    # Step 1: kumpulkan crops per kelas
    augmentor = build_augmentor()
    crops_by_class = collect_crops_by_class(augmentor, args.augments_per_crop)

    # Step 2: generate frame baru
    generate_augmented_frames(crops_by_class, output_images, output_labels)

    # Step 3: copy original
    copy_original_dataset(output_images, output_labels)

    # Distribusi sesudah
    print_class_distribution(output_labels, "SESUDAH (train_v4)")

    print("\n" + "=" * 60)
    print("  SELESAI!")
    print("=" * 60)
    print(f"\n  Output train set V4: {output_dir}")
    print("  Langkah selanjutnya:")
    print("  1. Zip folder dataset_yolo_split (termasuk train_v4/)")
    print("  2. Upload ke Colab dan jalankan train_colab_v4.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
