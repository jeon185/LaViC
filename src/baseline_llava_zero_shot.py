#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_item_meta(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_recall_at_k(recommended_ids, gt_items, k=1):
    hits = 0
    total = 0
    for rec_id, gts in zip(recommended_ids, gt_items):
        for gt in gts:
            if rec_id is not None and rec_id == gt:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0.0


def check_validity(file_path, model_key):
    candidates_key = f"candidates_{model_key}"
    recommended_key = f"recommended_{model_key}"

    total = 0
    valid = 0
    invalid_entries = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            recommended = data.get(recommended_key, None)
            candidates = data.get(candidates_key, [])
            total += 1

            if recommended in candidates:
                valid += 1
            else:
                invalid_entries.append({
                    "line_number": idx,
                    "recommended_id": recommended,
                    "candidates": candidates
                })

    validity = valid / total if total > 0 else 0.0
    return validity, invalid_entries, total


def prepare_candidate_info(candidates, item_meta):
    candidate_info = []
    for cid in candidates:
        title = item_meta.get(cid, {}).get("title", "No Title")
        candidate_info.append({
            "id": cid,
            "title": title
        })
    return candidate_info


def build_prompt_title_only(conversation_text, candidates_info):
    prompt = (
        "You are an AI assistant specialized in providing personalized product recommendations based on user conversations. "
        "You are given a conversation between a user seeking recommendation (denoted by <submission>) and other users providing comments (denoted by <comment>). "
        "You are also given a set of candidate products with their IDs and titles formatted as \"ID: title\". "
        "Among the candidates, recommend the most relevant product to the seeker. "
        "Only reply with its ID, and don't say anything else.\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Candidates:\n"
    )

    for candidate in candidates_info:
        cid = candidate["id"]
        title = candidate["title"]
        prompt += f"{cid}: {title}\n"

    prompt += "\nAssistant:"
    return prompt


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class RecommendationEvalDataset(Dataset):
    def __init__(self, data, item_meta, candidate_type):
        self.data = data
        self.item_meta = item_meta
        self.candidate_type = candidate_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        conversation_text = entry.get("context", "")
        gt_items = entry.get("gt_items", [])
        candidates = entry.get(self.candidate_type, [])

        candidate_info = prepare_candidate_info(candidates, self.item_meta)
        prompt = build_prompt_title_only(conversation_text, candidate_info)

        return {
            "prompt": prompt,
            "gt_items": gt_items,
            "entry_idx": idx,
            "candidates": candidates
        }


def collate_fn(batch):
    return batch


# ------------------------------------------------------------
# Main inference
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Zero-shot LLaVA title-only baseline for conversational recommendation")
    parser.add_argument("--base_model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--candidate_type", type=str, default="candidates_st")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--item_meta_path", type=str, default="./data/item2meta_train_amazon_home.json")
    parser.add_argument("--category", type=str, default="amazon_home")
    parser.add_argument("--output_dir", type=str, default="./out_baseline_llava_title_only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # run from notebook in LaViC root
    test_data_path = os.path.join("./data", args.category, "test.jsonl")

    item_meta = load_item_meta(args.item_meta_path)
    test_data = load_jsonl(test_data_path)

    test_dataset = RecommendationEvalDataset(
        data=test_data,
        item_meta=item_meta,
        candidate_type=args.candidate_type
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print("[INFO] Loading base LLaVA model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"

    test_results = []

    print("[INFO] Running zero-shot title-only baseline on test data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            prompts = [x["prompt"] for x in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            recommended_ids = []
            for txt in generated_texts:
                match = re.findall(r"\bB[A-Z0-9]{9}\b", txt.strip())
                recommended_ids.append(match[0][:10] if match else None)

            for i in range(len(batch)):
                test_results.append({
                    "entry_idx": batch[i]["entry_idx"],
                    "recommended_id": recommended_ids[i],
                    "response": generated_texts[i]
                })

    results_by_idx = {res["entry_idx"]: res for res in test_results}

    recommended_ids = []
    ground_truths = []

    model_key = "st" if args.candidate_type == "candidates_st" else "gpt_large"
    if model_key == "st":
        recommended_field = "recommended_st"
        response_field = "response_st"
    else:
        recommended_field = f"recommended_{model_key}"
        response_field = f"response_{model_key}"

    for idx, entry in enumerate(test_data):
        res = results_by_idx.get(idx)
        if res:
            entry[recommended_field] = res["recommended_id"]
            entry[response_field] = res["response"]
            recommended_ids.append(res["recommended_id"])
        else:
            recommended_ids.append(None)
        ground_truths.append(entry.get("gt_items", []))

    recall = evaluate_recall_at_k(recommended_ids, ground_truths, k=1)
    print(f"[Test] Recall@1: {recall:.4f}")

    out_file_name = f"test_results_{args.candidate_type}.jsonl"
    output_file_path = os.path.join(args.output_dir, out_file_name)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for entry in test_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"Test details saved to {output_file_path}")

    validity, invalid_entries, total_count = check_validity(output_file_path, model_key)
    print(f"Validity@1: {validity:.4f}, invalid entries: {len(invalid_entries)} / {total_count}")

    summary_file = os.path.join(args.output_dir, "results_summary.csv")
    csv_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow([
                "base_model_name", "candidate_type", "category",
                "recall@1", "validity@1", "output_file"
            ])
        writer.writerow([
            args.base_model_name,
            args.candidate_type,
            args.category,
            recall,
            validity,
            out_file_name
        ])

    print(f"Results summary updated: {summary_file}")
    print("Done.")


if __name__ == "__main__":
    main()