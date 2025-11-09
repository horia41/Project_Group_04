import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ======= CONFIG =======
SOURCE_DIR = Path("current_data")  # has Alternaria_2019/ and Alternaria_2022/, each with 0/ and 1/
OUTPUT_DIR = Path("DeepLearning_PlantDiseases-master/Scripts/PlantVillage")  # will contain train/, val/, test/
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42
USE_SYMLINKS = False  # set True if you prefer symlinks instead of copying (faster, saves disk)
# ======================

CLASS_MAPPING = {
    "0": "other (Not alternaria solani)",
    "1": "Alternaria solani"
}

YEAR_2019 = SOURCE_DIR / "2019"
YEAR_2022 = SOURCE_DIR / "2022"

def _ensure_clean_dirs():
    # make clean train/val/test class dirs
    for split in ["train", "val", "test"]:
        for cname in CLASS_MAPPING.values():
            d = OUTPUT_DIR / split / cname
            d.mkdir(parents=True, exist_ok=True)
            # clean existing files in case you re-run
            for f in d.iterdir():
                if f.is_file() or f.is_symlink():
                    f.unlink()

def _list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]

def _place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINKS:
        if dst.exists(): dst.unlink()
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)

def create_train_val_test():
    assert YEAR_2019.exists(), f"Missing {YEAR_2019}"
    assert YEAR_2022.exists(), f"Missing {YEAR_2022}"

    _ensure_clean_dirs()

    # -------- 2019 -> train/val (stratified per class) --------
    print("Processing 2019 -> train/val")
    for cls_id, cls_name in CLASS_MAPPING.items():
        src_dir = YEAR_2019 / cls_id
        imgs = _list_images(src_dir)
        print(f"  Class {cls_id} ({cls_name}): {len(imgs)} images")

        # stratified by class: we split within each class, then place
        train_imgs, val_imgs = train_test_split(
            imgs,
            train_size=TRAIN_SPLIT,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        print(f"    -> train: {len(train_imgs)} | val: {len(val_imgs)}")

        for p in train_imgs:
            _place(p, OUTPUT_DIR / "train" / cls_name / p.name)

        for p in val_imgs:
            _place(p, OUTPUT_DIR / "val" / cls_name / p.name)

    # -------- 2022 -> test (ALL) --------
    print("Processing 2022 -> test (ALL images)")
    total_2022 = 0
    for cls_id, cls_name in CLASS_MAPPING.items():
        src_dir = YEAR_2022 / cls_id
        imgs = _list_images(src_dir)
        print(f"  Class {cls_id} ({cls_name}): {len(imgs)} images")
        total_2022 += len(imgs)

        for p in imgs:
            _place(p, OUTPUT_DIR / "test" / cls_name / p.name)

    print("\nDataset split completed")
    print(f"Output root: {OUTPUT_DIR}")
    for split in ["train", "val", "test"]:
        print(f"  {split}/")
        for cname in CLASS_MAPPING.values():
            n = len(list((OUTPUT_DIR / split / cname).glob("*")))
            print(f"    - {cname}: {n} files")
    print(f"\n2019 -> train/val ({int(100*TRAIN_SPLIT)}%/{int(100*(1-TRAIN_SPLIT))}%), 2022 -> test (100%).")

if __name__ == "__main__":
    create_train_val_test()


# Processing 2019 -> train/val
# Class 0 (other (Not alternaria solani)): 1710 images
# -> train: 1368 | val: 342
# Class 1 (Alternaria solani): 2795 images
# -> train: 2236 | val: 559
# Processing 2022 -> test (ALL images)
# Class 0 (other (Not alternaria solani)): 2130 images
# Class 1 (Alternaria solani): 1027 images
#
# Dataset split completed
# Output root: DeepLearning_PlantDiseases-master/Scripts/PlantVillage
# train/
# - other (Not alternaria solani): 1368 files
# - Alternaria solani: 2236 files
# val/
# - other (Not alternaria solani): 342 files
# - Alternaria solani: 559 files
# test/
# - other (Not alternaria solani): 2130 files
# - Alternaria solani: 1027 files
#
# 2019 -> train/val (80%/19%), 2022 -> test (100%).