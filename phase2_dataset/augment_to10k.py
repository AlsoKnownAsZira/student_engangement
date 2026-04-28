"""
Balance semua kelas di train ke TARGET crops dengan augmentasi.

Input : phase2_dataset/crops_v6/train/{High,Low,Medium}/
Output: crops tambahan di folder yang sama, prefix "aug_"

Valid dan test TIDAK disentuh.
"""

import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

SEED   = 42
TARGET = 10000
random.seed(SEED)

ROOT      = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "phase2_dataset" / "crops_v6" / "train"
CLASSES   = ["High", "Low", "Medium"]


def count_images(folder: Path) -> int:
    return len(list(folder.glob("*.jpg")))


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


def balance_class(cls: str):
    cls_dir = TRAIN_DIR / cls
    current = count_images(cls_dir)
    needed  = TARGET - current

    print(f"  {cls:8s}: {current} → target {TARGET}  (tambah {needed})")

    if needed <= 0:
        print(f"           sudah >= {TARGET}, dilewati.")
        return

    sources = [p for p in cls_dir.glob("*.jpg") if not p.name.startswith("aug_")]

    src_pool = sources * (needed // len(sources) + 2)
    random.shuffle(src_pool)

    for i, src_path in enumerate(src_pool[:needed]):
        img = Image.open(src_path).convert("RGB")
        aug = augment_one(img)
        out_name = f"aug_{i:05d}_{src_path.stem}.jpg"
        aug.save(cls_dir / out_name, "JPEG", quality=95)

    print(f"           selesai → {count_images(cls_dir)} crops")


def main():
    print(f"Train dir : {TRAIN_DIR}")
    print(f"Target    : {TARGET} per kelas\n")

    for cls in CLASSES:
        balance_class(cls)

    print("\n=== HASIL AKHIR ===")
    for cls in CLASSES:
        print(f"  {cls:8s}: {count_images(TRAIN_DIR / cls)}")


if __name__ == "__main__":
    main()
