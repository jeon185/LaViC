
import os, json, requests
from io import BytesIO
from PIL import Image

def save_image(url, path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        Image.open(BytesIO(response.content)).save(path)
        return True
    except:
        return False

base = "/kaggle/working/ds8008-group8-lavic/data"
os.makedirs(f"{base}/valid_images", exist_ok=True)
count_save = count_fail = count_exist = 0
with open(f"{base}/item2meta_valid.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        image_name = entry.get("image_name", "")
        url = entry.get("image", "")
        if not url or not image_name:
            continue
        path = f"{base}/valid_images/{image_name}"
        if os.path.exists(path):
            count_exist += 1
        else:
            if save_image(url, path):
                count_save += 1
            else:
                count_fail += 1
print(f"Exist: {count_exist}, Saved: {count_save}, Failed: {count_fail}")
