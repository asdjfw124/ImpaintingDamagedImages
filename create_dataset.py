from pathlib import Path
import random
import shutil

dataset_directory = ""
IN_ROOT = Path(dataset_directory)
OUT_ROOT = Path(dataset_directory)
TRAIN_K   = 100      # images per class for train
EVAL_K    = 10       # images per class for eval
SEED      = 74       # RNG seed
OVERWRITE = True     # remove existing 'train'/'eval' first
DRY_RUN   = False    # print actions only

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def prepare_dir(p: Path, overwrite: bool):
    if p.exists() and overwrite:
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def safe_name(target_dir: Path, desired_name: str) -> Path:
    base = Path(desired_name).stem
    ext  = Path(desired_name).suffix
    candidate = target_dir / f"{base}{ext}"
    i = 1
    while candidate.exists():
        candidate = target_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate

def main():
    rng = random.Random(SEED)

    train_root = OUT_ROOT / "train"
    eval_root  = OUT_ROOT / "eval"

    prepare_dir(train_root, overwrite=OVERWRITE)
    prepare_dir(eval_root,  overwrite=OVERWRITE)

    class_dirs = [d for d in IN_ROOT.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class folders found in: {IN_ROOT}")

    for class_dir in sorted(class_dirs):
        imgs = list_images(class_dir)
        n = len(imgs)
        if n == 0:
            print(f"{class_dir.name}: no images, skipped.")
            continue

        rng.shuffle(imgs)

        k_train = min(TRAIN_K, n)
        train_imgs = imgs[:k_train]

        remaining = imgs[k_train:]
        k_eval = min(EVAL_K, len(remaining))
        eval_imgs = remaining[:k_eval]

        for src in train_imgs:
            prefixed = f"{class_dir.name}__{src.name}"
            dst = safe_name(train_root, prefixed)
            if DRY_RUN:
                print(f"[DRY] copy {src} -> {dst}")
            else:
                shutil.copy2(src, dst)

        for src in eval_imgs:
            prefixed = f"{class_dir.name}__{src.name}"
            dst = safe_name(eval_root, prefixed)
            if DRY_RUN:
                print(f"[DRY] copy {src} -> {dst}")
            else:
                shutil.copy2(src, dst)

        print(f"{class_dir.name}: train={len(train_imgs)} eval={len(eval_imgs)} (available={n})")

    print("Done.")

if __name__ == "__main__":
    main()
