"""
Build crops_v8: crop langsung dari dataset_smp → 2-class balanced.

Alur:
  1. Crop setiap bbox dari dataset_smp/{split}/images+labels
  2. Mapping: High+Medium → Engaged | Low → NotEngaged
  3. Train only: augment offline Low sampai match jumlah Engaged

Augmentasi ringan (~37% synthetic) karena Low sudah 6130, gap ke Engaged ~2268.
Valid/test: merge saja, TIDAK diaugment (evaluasi tetap jujur).

Output: phase2_dataset/crops_v8/{train,valid,test}/{Engaged,NotEngaged}/
"""

import random
import shutil
import collections
import tempfile
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

SEED      = 42
PADDING   = 0.10        # 10% padding relatif terhadap bbox
CLASS_MAP = {0: "High", 1: "Low", 2: "Medium"}

random.seed(SEED)

ROOT   = Path(__file__).resolve().parent.parent
SRC    = ROOT / "dataset_smp"
DST    = ROOT / "phase2_dataset" / "crops_v8"
SPLITS = ["train", "valid", "test"]


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_one(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.4:
        img = ImageEnhance.Sharpness(img).enhance(random.choice([0.5, 1.5, 2.0]))
    if random.random() < 0.4:
        img = img.rotate(random.uniform(-10, 10), expand=False, fillcolor=(128, 128, 128))
    return img


# ── Crop step ────────────────────────────────────────────────────────────────

def crop_split_to_dirs(split: str, out_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    """
    Crop semua bbox dari dataset_smp/{split} ke out_dirs per class.
    Kembalikan dict class_name → list[Path] file yang berhasil di-crop.
    """
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cropped: dict[str, list[Path]] = {c: [] for c in CLASS_MAP.values()}
    skipped = 0
    label_files = sorted(lbl_dir.glob("*.txt"))

    for i, lbl_path in enumerate(label_files, 1):
        if i % 200 == 0 or i == len(label_files):
            print(f"  [{split}] {i}/{len(label_files)} label diproses ...", end="\r")

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

        for idx, line in enumerate(lbl_path.read_text().splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in CLASS_MAP:
                continue

            cx, cy, bw, bh = map(float, parts[1:5])
            cx_px, cy_px = cx * W, cy * H
            bw_px, bh_px = bw * W, bh * H

            pad_x = PADDING * bw_px
            pad_y = PADDING * bh_px
            x1 = max(0, int(cx_px - bw_px / 2 - pad_x))
            y1 = max(0, int(cy_px - bh_px / 2 - pad_y))
            x2 = min(W, int(cx_px + bw_px / 2 + pad_x))
            y2 = min(H, int(cy_px + bh_px / 2 + pad_y))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_name = CLASS_MAP[cls_id]
            out_path = out_dirs[cls_name] / f"{lbl_path.stem}__b{idx}.jpg"
            img.crop((x1, y1, x2, y2)).save(out_path, "JPEG", quality=95)
            cropped[cls_name].append(out_path)

    print()
    if skipped:
        print(f"  [WARN] {skipped} label tidak ada pasangan gambarnya")
    return cropped


# ── Augment to target ────────────────────────────────────────────────────────

def augment_to_target(src_files: list[Path], dst_dir: Path, target: int):
    current = len(list(dst_dir.glob("*.jpg")))
    needed  = target - current
    if needed <= 0:
        print(f"    [skip] sudah {current} >= {target}")
        return

    pool = src_files * (needed // max(len(src_files), 1) + 2)
    random.shuffle(pool)

    for i, src in enumerate(pool[:needed]):
        aug = augment_one(Image.open(src).convert("RGB"))
        aug.save(dst_dir / f"aug_{i:05d}_{src.name}", "JPEG", quality=95)

    print(f"    Augmented  : +{needed} crops (Low → NotEngaged)")


# ── Main ─────────────────────────────────────────────────────────────────────

def build_split(split: str):
    tmp_high   = DST / "_tmp" / split / "High"
    tmp_low    = DST / "_tmp" / split / "Low"
    tmp_medium = DST / "_tmp" / split / "Medium"

    dst_engaged    = DST / split / "Engaged"
    dst_notengaged = DST / split / "NotEngaged"
    dst_engaged.mkdir(parents=True, exist_ok=True)
    dst_notengaged.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split.upper()}] Cropping dari dataset_smp ...")
    cropped = crop_split_to_dirs(split, {
        "High":   tmp_high,
        "Low":    tmp_low,
        "Medium": tmp_medium,
    })

    # Salin ke 2-class folder
    for f in cropped["High"] + cropped["Medium"]:
        shutil.copy2(f, dst_engaged / f.name)
    for f in cropped["Low"]:
        shutil.copy2(f, dst_notengaged / f.name)

    n_engaged = len(cropped["High"]) + len(cropped["Medium"])

    if split == "train":
        print(f"    Engaged (High+Medium) : {n_engaged}")
        print(f"    NotEngaged (Low orig) : {len(cropped['Low'])}")
        augment_to_target(cropped["Low"], dst_notengaged, target=n_engaged)

    n_eng_final = len(list(dst_engaged.glob("*.jpg")))
    n_not_final = len(list(dst_notengaged.glob("*.jpg")))
    total = n_eng_final + n_not_final

    print(f"    Engaged    : {n_eng_final:5d}  ({100*n_eng_final/total:.1f}%)")
    print(f"    NotEngaged : {n_not_final:5d}  ({100*n_not_final/total:.1f}%)")
    print(f"    Total      : {total}")


def main():
    if DST.exists():
        print(f"[WARN] {DST} sudah ada. Hapus dulu jika ingin rebuild:\n  rm -rf {DST}")
        return

    print(f"Source : {SRC}")
    print(f"Output : {DST}")
    print("Mapping: High+Medium → Engaged | Low → NotEngaged")
    print("Augment offline: HANYA Low di train (sampai match Engaged)\n")

    for split in SPLITS:
        build_split(split)

    # Bersihkan folder tmp
    tmp = DST / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
        print("\n[INFO] Folder tmp dihapus.")

    print(f"\nSelesai. crops_v8 siap di-zip dan upload ke Kaggle.")
    print(f"Lokasi: {DST}")


if __name__ == "__main__":
    main()
