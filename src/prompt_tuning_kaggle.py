#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Stage 2: Recommendation Prompt Tuning — Kaggle T4 Adaptation
- Loads distilled vision LoRA adapter from Stage 1
- Applies new LoRA to language model side
- 4-bit QLoRA, batch=1, grad_accum=4, single GPU
"""

import argparse
import csv
import json
import os
import random
import re

import pytorch_lightning as pl
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from bitsandbytes.optim import PagedAdamW8bit

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]

def load_item_meta(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

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
    total = valid = 0
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
                invalid_entries.append({"line_number": idx, "recommended_id": recommended, "candidates": candidates})
    return (valid / total if total > 0 else 0.0), invalid_entries, total

def prepare_candidate_info(candidates, item_meta, image_dir, default_image):
    candidate_info = []
    for cid in candidates:
        title = item_meta.get(cid, {}).get("title", "No Title")
        image_path = os.path.join(image_dir, f"{cid}_0.jpg")
        image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else default_image
        candidate_info.append({"id": cid, "title": title, "image": image})
    return candidate_info

def build_prompt(conversation_text, candidates_info):
    # Truncate conversation to save memory
    if len(conversation_text) > 300:
        conversation_text = conversation_text[:300] + "..."
    prompt = (
        "You are an AI assistant for product recommendations. "
        "Given the conversation and candidates, recommend the most relevant product ID. "
        "Only reply with the ID.\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Candidates:\n"
    )
    for candidate in candidates_info:
        prompt += f"{candidate['id']}: {candidate['title']}\n"
        prompt += "".join(IMAGE_TOKENS) + "\n"
    prompt += "\nAssistant:"
    return prompt


def get_llava_core(model):
    seen = set()
    queue = [model]
    while queue:
        cand = queue.pop(0)
        if cand is None:
            continue
        obj_id = id(cand)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if (hasattr(cand, "vision_tower") and hasattr(cand, "multi_modal_projector") and hasattr(cand, "language_model")):
            return cand
        for attr in ["model", "base_model"]:
            if hasattr(cand, attr):
                nxt = getattr(cand, attr)
                if nxt is not None:
                    queue.append(nxt)
    raise AttributeError("Could not find LLaVA core.")


class RecommendationDataset(Dataset):
    def __init__(self, data, item_meta, image_dir, candidate_type, default_image, is_training=True):
        self.data = data
        self.item_meta = item_meta
        self.image_dir = image_dir
        self.candidate_type = candidate_type
        self.default_image = default_image
        self.is_training = is_training
        if is_training:
            self.index_mapping = [
                (entry_idx, gt_idx)
                for entry_idx, entry in enumerate(data)
                for gt_idx in range(len(entry.get("gt_items", [])))
            ]
        else:
            self.index_mapping = [(entry_idx, None) for entry_idx in range(len(data))]

    def __len__(self): return len(self.index_mapping)

    def __getitem__(self, idx):
        entry_idx, gt_idx = self.index_mapping[idx]
        entry = self.data[entry_idx]
        conversation_text = entry.get("context", "")
        gt_items = entry.get("gt_items", [])
        candidates = list(entry.get(self.candidate_type, []))
        if self.is_training:
            gt_item = gt_items[gt_idx]
            if gt_item not in candidates and len(candidates) > 0:
                candidates[random.randint(0, len(candidates) - 1)] = gt_item
            target_text = gt_item
        else:
            target_text = ""
        # Truncate to 5 candidates to fit in T4 VRAM
        candidates = candidates[:5]
        candidate_info = prepare_candidate_info(candidates, self.item_meta, self.image_dir, self.default_image)
        prompt = build_prompt(conversation_text, candidate_info)
        return {
            "prompt": prompt,
            "images": [c["image"] for c in candidate_info],
            "target_text": target_text,
            "gt_items": gt_items,
            "entry_idx": entry_idx
        }


class DataCollatorForLLaVA:
    def __init__(self, processor, tokenizer, max_length):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token_ids = [tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS]

    def __call__(self, batch):
        prompts = [item["prompt"] for item in batch]
        target_texts = [item["target_text"] for item in batch]
        images_per_sample = [item["images"] for item in batch]
        full_prompts = [p + t for p, t in zip(prompts, target_texts)]

        tokenized_prompts = self.tokenizer(prompts, max_length=self.max_length, padding="longest", truncation=True, return_tensors="pt")
        tokenized_full = self.tokenizer(full_prompts, max_length=self.max_length, padding="longest", truncation=True, return_tensors="pt")

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = input_ids.clone()
        for i, plen in enumerate([len(x) for x in tokenized_prompts["input_ids"]]):
            labels[i, :plen] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            for tid in self.image_token_ids:
                positions = (input_ids[b_idx] == tid).nonzero(as_tuple=False).squeeze(-1)
                image_token_mask[b_idx, positions] = True

        all_images = [img for imgs in images_per_sample for img in imgs]
        images_tensor = self.processor.image_processor(all_images, return_tensors="pt")["pixel_values"] if all_images else None
        images_per_sample_lengths = [5 * len(imgs) for imgs in images_per_sample]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_tensor,
            "image_token_mask": image_token_mask,
            "images_per_sample_lengths": images_per_sample_lengths
        }


class LLaVAModel(pl.LightningModule):
    def __init__(self, model, processor, tokenizer, args):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.args = args
        self.save_hyperparameters(ignore=["model", "processor", "tokenizer"])
        self.data_collator = DataCollatorForLLaVA(processor, tokenizer, max_length=args.max_length)
        self.test_results = []

    def forward(self, input_ids, attention_mask, images, image_token_mask, images_per_sample_lengths, labels=None):
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_token_mask = image_token_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        core = get_llava_core(self.model)
        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()

        if images is not None:
            images = images.to(device, dtype=torch.float16)
            B_text = input_ids.size(0)
            B_prime = images.shape[0]
            num_views = images.shape[1]
            C, H, W = images.shape[2], images.shape[3], images.shape[4]

            images_reshaped = images.view(B_prime * num_views, C, H, W)
            with torch.no_grad():
                vision_outputs = core.vision_tower(images_reshaped)

            cls_states = vision_outputs.last_hidden_state[:, 0, :].view(B_prime, num_views, -1)
            candidate_count = B_prime // B_text
            cls_states = cls_states.view(B_text, candidate_count, num_views, -1)
            cls_states = core.multi_modal_projector(cls_states)

            for b_idx in range(B_text):
                image_positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                if image_positions.numel() == 0:
                    continue
                pos_count = min(len(image_positions), candidate_count * num_views)
                for c in range(candidate_count):
                    for i in range(num_views):
                        idx_token = c * num_views + i
                        if idx_token >= pos_count:
                            break
                        inputs_embeds[b_idx, image_positions[idx_token].item(), :] = cls_states[b_idx, c, i, :]

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        # Check labels have valid tokens
        valid_tokens = (inputs["labels"] != -100).sum().item()
        if valid_tokens == 0:
            return None
        outputs = self(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            images=inputs["images"], image_token_mask=inputs["image_token_mask"],
            images_per_sample_lengths=inputs["images_per_sample_lengths"], labels=inputs["labels"]
        )
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            images=inputs["images"], image_token_mask=inputs["image_token_mask"],
            images_per_sample_lengths=inputs["images_per_sample_lengths"], labels=inputs["labels"]
        )
        self.log("val_loss", outputs.loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return {"val_loss": outputs.loss}

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs = self.data_collator(batch)
            device = next(self.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            image_token_mask = inputs["image_token_mask"].to(device)
            images = inputs["images"]

            core = get_llava_core(self.model)
            inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()

            if images is not None:
                images = images.to(device, dtype=torch.float16)
                B_text = input_ids.size(0)
                B_prime = images.shape[0]
                num_views = images.shape[1]
                C, H, W = images.shape[2], images.shape[3], images.shape[4]
                images_reshaped = images.view(B_prime * num_views, C, H, W)
                vision_outputs = core.vision_tower(images_reshaped)
                cls_states = vision_outputs.last_hidden_state[:, 0, :].view(B_prime, num_views, -1)
                candidate_count = B_prime // B_text
                cls_states = cls_states.view(B_text, candidate_count, num_views, -1)
                cls_states = core.multi_modal_projector(cls_states)
                for b_idx in range(B_text):
                    image_positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                    if image_positions.numel() == 0:
                        continue
                    pos_count = min(len(image_positions), candidate_count * num_views)
                    for c in range(candidate_count):
                        for i in range(num_views):
                            idx_token = c * num_views + i
                            if idx_token >= pos_count:
                                break
                            inputs_embeds[b_idx, image_positions[idx_token].item(), :] = cls_states[b_idx, c, i, :]

            generated_ids = self.model.generate(
                input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                max_new_tokens=10, num_beams=1, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            recommended_ids = []
            for txt in generated_texts:
                match = re.findall(r"\bB[A-Z0-9]{9}\b", txt.strip())
                recommended_ids.append(match[0][:10] if match else None)

            gt_items_list = [b["gt_items"] for b in batch]
            entry_idxs = [b["entry_idx"] for b in batch]
            for i in range(len(batch)):
                self.test_results.append({"entry_idx": entry_idxs[i], "recommended_id": recommended_ids[i], "response": generated_texts[i]})

            recall = evaluate_recall_at_k(recommended_ids, gt_items_list, k=1)
            self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
            return {"test_recall": recall}

    def configure_optimizers(self):
        return PagedAdamW8bit(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )


def find_llm_linear_layer_names(llm_module, prefix="language_model"):
    linear_names = []
    for name, module in llm_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(f"{prefix}.{name}" if prefix else name)
    return linear_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--vision_adapter_dir", type=str, default="/kaggle/working/vision_adapter_fixed")
    parser.add_argument("--candidate_type", type=str, default="candidates_st")
    parser.add_argument("--finetune_output_dir", type=str, default="/kaggle/working/out_finetuned")
    parser.add_argument("--item_meta_path", type=str, default="/kaggle/working/ds8008-group8-lavic/data/item2meta_train.json")
    parser.add_argument("--image_dir", type=str, default="/kaggle/working/ds8008-group8-lavic/data/train_images")
    parser.add_argument("--category", type=str, default="amazon_home")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args([])

    seed_everything(args.seed)
    os.makedirs(args.finetune_output_dir, exist_ok=True)

    # Data paths
    data_root = "/kaggle/working/ds8008-group8-lavic/data"
    args.train_data_path = f"{data_root}/{args.category}/train.jsonl"
    args.val_data_path = f"{data_root}/{args.category}/valid.jsonl"
    args.test_data_path = f"{data_root}/{args.category}/test.jsonl"

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )

    # Load base model
    print("[INFO] Loading base model...")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model_name, quantization_config=bnb_config,
        device_map={"": 0}, torch_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    processor.tokenizer.padding_side = "right"
    tokenizer = processor.tokenizer

    # Add special tokens BEFORE loading adapter (must match Stage 1 vocab size)
    tokenizer.add_special_tokens({"additional_special_tokens": IMAGE_TOKENS})
    tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))

    # Load vision LoRA adapter from Stage 1
    print("[INFO] Loading Stage 1 vision LoRA adapter...")
    base_model = PeftModel.from_pretrained(base_model, args.vision_adapter_dir)

    # Prepare for kbit training
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.gradient_checkpointing_enable()
    
    # Freeze vision tower completely in Stage 2 (only LLM trains)
    core_check = base_model
    for attr in ["model", "base_model"]:
        if hasattr(core_check, attr):
            core_check = getattr(core_check, attr)
            if hasattr(core_check, "vision_tower"):
                for param in core_check.vision_tower.parameters():
                    param.requires_grad = False
                print("[INFO] Vision tower frozen for Stage 2")
                break

    # Apply new LoRA to language model
    print("[INFO] Applying Stage 2 LoRA to language model...")
    core = get_llava_core(base_model)
    lm_linear_names = find_llm_linear_layer_names(core.language_model, prefix="model.language_model")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.1, target_modules=lm_linear_names
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("[INFO] Loading data...")
    item_meta = load_item_meta(args.item_meta_path)
    train_data = load_jsonl(args.train_data_path)
    val_data = load_jsonl(args.val_data_path)
    test_data = load_jsonl(args.test_data_path)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    default_image = Image.new("RGB", (336, 336), color=(255, 255, 255))

    train_dataset = RecommendationDataset(train_data, item_meta, args.image_dir, args.candidate_type, default_image, is_training=True)
    val_dataset = RecommendationDataset(val_data, item_meta, args.image_dir, args.candidate_type, default_image, is_training=False)
    test_dataset = RecommendationDataset(test_data, item_meta, args.image_dir, args.candidate_type, default_image, is_training=False)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    collate_fn = lambda x: x
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    llava_model = LLaVAModel(model, processor, tokenizer, args)

    # Custom LoRA-only checkpointing callback
    import glob

    class LoRACheckpoint(pl.Callback):
        def __init__(self, dirpath, every_n_steps=500):
            self.dirpath = dirpath
            self.every_n_steps = every_n_steps

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step > 0 and step % self.every_n_steps == 0:
                save_path = os.path.join(self.dirpath, f"lora_adapter_step{step}")
                pl_module.model.save_pretrained(save_path)
                print(f"[INFO] LoRA adapter saved at step {step} to {save_path}")

    lora_ckpt_callback = LoRACheckpoint(args.finetune_output_dir, every_n_steps=500)

    # Check for existing LoRA adapter to resume from
    existing_adapters = sorted(glob.glob(f"{args.finetune_output_dir}/lora_adapter_step*"))
    resume_step = 0
    if existing_adapters:
        latest = existing_adapters[-1]
        resume_step = int(latest.split("step")[-1])
        print(f"[INFO] Loading LoRA adapter from step {resume_step}: {latest}")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as safe_load
        adapter_file = os.path.join(latest, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(latest, "adapter_model.bin")
        if os.path.exists(adapter_file):
            if adapter_file.endswith(".safetensors"):
                state = safe_load(adapter_file)
            else:
                state = torch.load(adapter_file, map_location="cpu", weights_only=False)
            set_peft_model_state_dict(model, state)
            print(f"[INFO] Resumed LoRA weights from step {resume_step}")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu", devices=1,
        callbacks=[lora_ckpt_callback],
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
    )

    print("[INFO] Starting Stage 2 training...")
    trainer.fit(llava_model, train_loader, val_loader)

    print("[INFO] Running test...")
    trainer.test(llava_model, dataloaders=test_loader)

    # Compile results
    results_by_idx = {res["entry_idx"]: res for res in llava_model.test_results}
    recommended_ids = []
    ground_truths = []
    for idx, entry in enumerate(test_data):
        result = results_by_idx.get(idx)
        recommended_ids.append(result["recommended_id"] if result else None)
        ground_truths.append(entry.get("gt_items", []))

    recall = evaluate_recall_at_k(recommended_ids, ground_truths, k=1)
    print(f"[Test] HR@1: {recall:.4f}")

    model_key = "st" if args.candidate_type == "candidates_st" else "gpt_large"
    recommended_field = f"recommended_{model_key}"
    response_field = f"response_{model_key}"

    for idx, entry in enumerate(test_data):
        res = results_by_idx.get(idx)
        if res:
            entry[recommended_field] = res["recommended_id"]
            entry[response_field] = res["response"]

    out_file_name = f"test_results_{args.candidate_type}.jsonl"
    output_file_path = os.path.join(args.finetune_output_dir, out_file_name)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for entry in test_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"[INFO] Test results saved to {output_file_path}")

    validity, invalid_entries, total_count = check_validity(output_file_path, model_key)
    print(f"[Test] VR: {validity:.4f}, invalid: {len(invalid_entries)}/{total_count}")

    # Save LoRA adapter
    trained_lora_path = os.path.join(args.finetune_output_dir, "trained_lora_adapter")
    model.save_pretrained(trained_lora_path)
    print(f"[INFO] Stage 2 LoRA adapter saved to {trained_lora_path}")

    # Save summary
    summary_file = os.path.join(args.finetune_output_dir, "results_summary.csv")
    csv_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["candidate_type", "lr", "num_epochs", "HR@1", "VR", "output_file"])
        writer.writerow([args.candidate_type, args.lr, args.num_epochs, recall, validity, out_file_name])
    print(f"[INFO] Summary saved to {summary_file}")


if __name__ == "__main__":
    import torch.serialization as _ts
    _ts.add_safe_globals([argparse.Namespace])
    main()
