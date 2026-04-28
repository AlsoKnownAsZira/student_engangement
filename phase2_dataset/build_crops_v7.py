"""
Build crops_v7: 2-class dari crops_v6, balanced di train.

Mapping:
  Engaged    = High + Medium  (original saja, tidak di-augment offline)
  NotEngaged = Low            (original → augment offline sampai match Engaged)

Split valid dan test: hanya merge, TIDAK diaugment (evaluasi tetap jujur).

Output: phase2_dataset/crops_v7/{train,valid,test}/{Engaged,NotEngaged}/
"""

import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

SEED = 42
random.seed(SEED)

ROOT    = Path(__file__).resolve().parent.parent
SRC     = ROOT / "phase2_dataset" / "crops_v6"
DST     = ROOT / "phase2_dataset" / "crops_v7"
SPLITS  = ["train", "valid", "test"]


def original_files(folder: Path) -> list[Path]:
    return [p for p in folder.glob("*.jpg") if not p.name.startswith("aug_")]


def augment_one(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Sharpness(img).enhance(random.choice([0.5, 1.5, 2.0]))
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-10, 10), expand=False, fillcolor=(128, 128, 128))
    return img


def copy_files(files: list[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)


def augment_to_target(src_files: list[Path], dst_dir: Path, target: int):
    """Augment src_files sampai total di dst_dir = target."""
    current = len(list(dst_dir.glob("*.jpg")))
    needed  = target - current
    if needed <= 0:
        return

    pool = src_files * (needed // len(src_files) + 2)
    random.shuffle(pool)

    for i, src in enumerate(pool[:needed]):
        img = Image.open(src).convert("RGB")
        aug = augment_one(img)
        aug.save(dst_dir / f"aug_{i:05d}_{src.name}", "JPEG", quality=95)


def build_split(split: str):
    src_high   = original_files(SRC / split / "High")
    src_low    = original_files(SRC / split / "Low")
    src_medium = original_files(SRC / split / "Medium")

    dst_engaged    = DST / split / "Engaged"
    dst_notengaged = DST / split / "NotEngaged"

    # Engaged = High + Medium (original, tanpa augment offline)
    copy_files(src_high,   dst_engaged)
    copy_files(src_medium, dst_engaged)
    n_engaged = len(src_high) + len(src_medium)

    # NotEngaged = Low original
    copy_files(src_low, dst_notengaged)

    # Augment offline hanya di train supaya balance
    if split == "train":
        augment_to_target(src_low, dst_notengaged, target=n_engaged)

    n_engaged_final    = len(list(dst_engaged.glob("*.jpg")))
    n_notengaged_final = len(list(dst_notengaged.glob("*.jpg")))
    total = n_engaged_final + n_notengaged_final

    print(f"\n  [{split.upper()}]")
    print(f"    Engaged    : {n_engaged_final:5d}  ({100*n_engaged_final/total:.1f}%)")
    print(f"    NotEngaged : {n_notengaged_final:5d}  ({100*n_notengaged_final/total:.1f}%)")
    print(f"    Total      : {total}")
    if split == "train":
        print(f"    Low aug    : +{n_notengaged_final - len(src_low)} crops")


def main():
    if DST.exists():
        print(f"[WARN] {DST} sudah ada, hapus dulu jika ingin rebuild.")
        return

    print(f"Source : {SRC}")
    print(f"Output : {DST}")
    print(f"\nMapping: High+Medium → Engaged | Low → NotEngaged")
    print(f"Augment offline: HANYA Low di train (sampai match Engaged)")

    for split in SPLITS:
        build_split(split)

    print("\n\nSelesai. crops_v7 siap di-zip dan upload ke Kaggle.")


if __name__ == "__main__":
    main()
