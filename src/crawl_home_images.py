
import os, json, requests
from io import BytesIO
from PIL import Image

def save_image(url, path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        Image.open(BytesIO(response.content)).save(path)
        return True
    except Exception as e:
        return False

base = "/kaggle/working/ds8008-group8-lavic/data"
home_item_ids = set()
for split in ["train", "valid", "test"]:
    path = f"{base}/amazon_home/{split}.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                for item in entry.get("gt_items", []):
                    home_item_ids.add(item)
                for key in ["candidates_st", "candidates_gpt_large"]:
                    for item in entry.get(key, []):
                        home_item_ids.add(item)
print(f"Home item IDs: {len(home_item_ids)}")

os.makedirs(f"{base}/train_images", exist_ok=True)
with open(f"{base}/item2meta_train.json") as f:
    data = json.load(f)

count_save = count_fail = count_exist = count_skip = 0
for item_id, details in data.items():
    if item_id not in home_item_ids:
        count_skip += 1
        continue
    if "images" in details and len(details["images"]) > 0:
        url = details["images"][0].get("large", "")
        if not url:
            continue
        path = f"{base}/train_images/{item_id}_0.jpg"
        if os.path.exists(path):
            count_exist += 1
        else:
            if save_image(url, path):
                count_save += 1
            else:
                count_fail += 1

print(f"Exist: {count_exist}, Saved: {count_save}, Failed: {count_fail}, Skipped: {count_skip}")
