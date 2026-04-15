import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

TRAIN_JSONL = r"./amazon_home/train.jsonl"
TRAIN_IMAGE_DIR = r"./train_images"
OUTPUT_DIR = r"./amazon_home_train_images_subset"

MAX_IMAGES_PER_ITEM = 2  #<-------------change this for different runs

USE_HARDLINK = False

# =========================================================
# STEP 1: collect ASINs from train.jsonl
# =========================================================
def collect_asins_from_jsonl(jsonl_path):
    asin_set = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ["gt_items", "candidates_gpt_large", "candidates_st", "context_items"]:
                values = row.get(key, [])
                if isinstance(values, list):
                    for x in values:
                        if isinstance(x, str) and x:
                            asin_set.add(x)
    return asin_set

# =========================================================
# STEP 2: parse filenames efficiently
# =========================================================
def parse_image_filename(filename):
    stem, ext = os.path.splitext(filename)
    if "_" not in stem:
        return None, None, None
    asin, idx_str = stem.rsplit("_", 1)
    if not asin:
        return None, None, None
    if not idx_str.isdigit():
        return None, None, None
    return asin, int(idx_str), ext.lower()

# =========================================================
# STEP 3: build subset
# =========================================================
def make_subset(train_image_dir,output_dir,valid_asins,max_images_per_item=2,use_hardlink=False):
    src_dir = Path(train_image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched = defaultdict(list)
    with os.scandir(src_dir) as it:
        for entry in it:
            
            if not entry.is_file():
                continue
            asin, idx, ext = parse_image_filename(entry.name)
            if asin is None:
                continue

            if asin in valid_asins:
                matched[asin].append((idx, entry.path, entry.name))

    copied_count = 0
    item_count = 0

    for asin, files in matched.items():
        # keep first N by numeric suffix
        files.sort(key=lambda x: x[0])
        selected = files[:max_images_per_item]
        for _, src_path, fname in selected:
            dst_path = out_dir / fname
            if dst_path.exists():
                continue
                
            if use_hardlink:
                os.link(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            copied_count += 1

        item_count += 1

    return item_count, copied_count, matched

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Reading train.jsonl...")
    asins = collect_asins_from_jsonl(TRAIN_JSONL)
    print(f"Unique ASINs found in train.jsonl: {len(asins):,}")

    print("Building subset folder...")
    item_count, copied_count, matched = make_subset(
        train_image_dir=TRAIN_IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        valid_asins=asins,
        max_images_per_item=MAX_IMAGES_PER_ITEM,
        use_hardlink=USE_HARDLINK
    )

    print("\nDone.")
    print(f"Items with at least one matching image: {item_count:,}")
    print(f"Images copied/linked: {copied_count:,}")
    print(f"Subset folder: {OUTPUT_DIR}")
    print(f"Max images kept per item: {MAX_IMAGES_PER_ITEM}")
    print(f"Used hardlinks: {USE_HARDLINK}")