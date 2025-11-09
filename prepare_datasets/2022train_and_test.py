import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("../current_data")  # has 2019/ and 2022/
OUTPUT_DIR = Path("../DeepLearning_PlantDiseases-master/Scripts/PlantVillage_2022")
TRAIN_SPLIT = 0.62
RANDOM_STATE = 42
USE_SYMLINKS = False  # set True if you prefer symlinks instead of copying (faster, saves disk)

CLASS_MAPPING = {
    "0": "other (Not alternaria solani)",
    "1": "Alternaria solani"
}

YEAR_2022 = SOURCE_DIR / "2022"

def _ensure_clean_dirs():
    # make clean train/test class dirs
    for split in ["train", "test"]:
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

def create_train_test_split(): # Renamed function
    assert YEAR_2022.exists(), f"Missing {YEAR_2022}"

    _ensure_clean_dirs()

    # -------- 2022 -> train/test (stratified 62/38 split) --------
    print(f"Processing 2022 -> train/test ({int(TRAIN_SPLIT*100)}% / {int((1-TRAIN_SPLIT)*100)}%)")
    for cls_id, cls_name in CLASS_MAPPING.items():
        src_dir = YEAR_2022 / cls_id
        imgs = _list_images(src_dir)
        print(f"  Class {cls_id} ({cls_name}): {len(imgs)} images")

        train_imgs, test_imgs = train_test_split(
            imgs,
            train_size=TRAIN_SPLIT,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        print(f"    -> train: {len(train_imgs)} | test: {len(test_imgs)}")

        for p in train_imgs:
            _place(p, OUTPUT_DIR / "train" / cls_name / p.name)

        for p in test_imgs:
            _place(p, OUTPUT_DIR / "test" / cls_name / p.name)

    print("\nDataset split completed")
    print(f"Output root: {OUTPUT_DIR}")
    for split in ["train", "test"]:
        print(f"  {split}/")
        for cname in CLASS_MAPPING.values():
            n = len(list((OUTPUT_DIR / split / cname).glob("*")))
            print(f"    - {cname}: {n} files")

    print(f"\n2022 -> train/test ({int(100*TRAIN_SPLIT)}%/{int(100*(1-TRAIN_SPLIT))}%) split.")

if __name__ == "__main__":
    create_train_test_split()