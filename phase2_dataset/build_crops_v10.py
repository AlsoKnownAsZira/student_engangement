"""
Build crops_v10 dari crops_v7 dengan SESSION-BASED MULTI-VAL split.

Perubahan vs V7:
- Val multi-session (4mar_0917 + 6mar_0959) -- bukan 1 sesi
- Train: 4 sesi (2mar_0830, 3mar_1102, 5mar_0824, 5mar_1024)
- Test tetap = 2mar_0906 (apple-to-apple dengan V7)
- DROP semua file augmented offline (`_aug` di nama)
- Class imbalance ditangani di training (class weights / sampler), bukan offline aug

Hasil estimasi crops_v10:
  Train     : Engaged ~8042, NotEngaged ~3502  (rasio ~70:30, original only)
  Valid     : Engaged ~2116, NotEngaged ~1389  (2 sesi, 60:40)
  Test      : Engaged  2628, NotEngaged 1610   (1 sesi, sama V7)

Jalankan dari root project:
  python phase2_dataset/build_crops_v10.py
"""

from pathlib import Path
import shutil
import re
import argparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "crops_v7"
DST = ROOT / "crops_v10"
CLASSES = ["Engaged", "NotEngaged"]

SESSION_PAT = re.compile(r"(Kelas\d+_\d+[a-zA-Z]+_\d+)", re.IGNORECASE)
AUG_PAT = re.compile(r"_aug|aug_", re.IGNORECASE)

SPLIT_BY_SESSION = {
    "Kelas9_2mar_0830": "train",
    "Kelas9_3mar_1102": "train",
    "Kelas9_5mar_0824": "train",
    "Kelas9_5mar_1024": "train",
    "Kelas9_4mar_0917": "valid",
    "Kelas9_6mar_0959": "valid",
    "Kelas9_2mar_0906": "test",
}


def session_of(name: str) -> str:
    m = SESSION_PAT.search(name)
    return m.group(1) if m else ""


def is_augmented(name: str) -> bool:
    return bool(AUG_PAT.search(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symlink", action="store_true",
                    help="Pakai symlink (cepat, hemat disk). Default: copy.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"Source tidak ada: {SRC}")

    if DST.exists() and not args.dry_run:
        print(f"[WARN] {DST} sudah ada — hapus dulu? (Ctrl+C untuk batal)")
        input("Tekan Enter untuk hapus & rebuild ...")
        shutil.rmtree(DST)

    # Buat struktur target
    if not args.dry_run:
        for split in ["train", "valid", "test"]:
            for cls in CLASSES:
                (DST / split / cls).mkdir(parents=True, exist_ok=True)

    stats = {(s, c): {"orig": 0, "skipped_aug": 0, "skipped_unknown": 0}
             for s in ["train", "valid", "test"] for c in CLASSES}

    for src_split in ["train", "valid", "test"]:
        for cls in CLASSES:
            src_dir = SRC / src_split / cls
            if not src_dir.exists():
                continue
            for img in src_dir.iterdir():
                if not img.is_file():
                    continue
                if is_augmented(img.name):
                    # Drop semua synthetic
                    sess = session_of(img.name)
                    target_split = SPLIT_BY_SESSION.get(sess)
                    if target_split:
                        stats[(target_split, cls)]["skipped_aug"] += 1
                    continue
                sess = session_of(img.name)
                target_split = SPLIT_BY_SESSION.get(sess)
                if target_split is None:
                    stats[(src_split, cls)]["skipped_unknown"] += 1
                    continue
                dst_path = DST / target_split / cls / img.name
                if not args.dry_run:
                    if args.symlink:
                        try:
                            dst_path.symlink_to(img.resolve())
                        except FileExistsError:
                            pass
                    else:
                        shutil.copy2(img, dst_path)
                stats[(target_split, cls)]["orig"] += 1

    # Summary
    print("\n" + "=" * 70)
    print("HASIL BUILD CROPS_V10")
    print("=" * 70)
    print(f"{'Split':<8} {'Class':<12} {'Original':>10} {'SkippedAug':>12} {'Unknown':>10}")
    print("-" * 70)
    for split in ["train", "valid", "test"]:
        for cls in CLASSES:
            v = stats[(split, cls)]
            print(f"{split:<8} {cls:<12} {v['orig']:>10} {v['skipped_aug']:>12} {v['skipped_unknown']:>10}")
        # Totals per split
        eng = stats[(split, "Engaged")]["orig"]
        notg = stats[(split, "NotEngaged")]["orig"]
        tot = eng + notg
        ratio = (eng / tot * 100) if tot else 0
        print(f"{'  ':<8} {'TOTAL':<12} {tot:>10}     ({ratio:.1f}% Engaged)")
        print("-" * 70)

    print(f"\nTarget: {DST}")
    if args.dry_run:
        print("(dry-run, tidak ada file yang ditulis)")
    else:
        mode = "symlink" if args.symlink else "copy"
        print(f"Mode  : {mode}")
        print("\nSiap dipakai untuk training V10.")


if __name__ == "__main__":
    main()
