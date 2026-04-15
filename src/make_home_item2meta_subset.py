import json

TRAIN_JSONL = "./amazon_home/train.jsonl"
FULL_META = "./item2meta_train.json"
OUT_META = "./item2meta_train_amazon_home.json"

def collect_asins(jsonl_path):
    asins = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ["gt_items", "candidates_gpt_large", "candidates_st", "context_items"]:
                vals = row.get(key, [])
                if isinstance(vals, list):
                    for x in vals:
                        if isinstance(x, str) and x:
                            asins.add(x)
    return asins

if __name__ == "__main__":
    used_asins = collect_asins(TRAIN_JSONL)
    with open(FULL_META, "r", encoding="utf-8") as f:
        full_meta = json.load(f)
    sub_meta = {k: v for k, v in full_meta.items() if k in used_asins}
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(sub_meta, f)
    print(f"ASINs in train.jsonl: {len(used_asins):,}")
    print(f"ASINs written to subset meta: {len(sub_meta):,}")
    print(f"Saved to: {OUT_META}")