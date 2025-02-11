#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Code: Recommendation Prompt Tuning
----------------------------------------------------------------------
This script fine-tunes a pretrained LLaVA model on a recommendation task,
applying LoRA to the language model side. We always assume images
are used in the prompt (5 sub-image approach).
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
from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


##############################################################
# LLaVA v1.6: 5 image tokens
##############################################################
IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]


##############################################################
# Utility Functions
##############################################################
def load_item_meta(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def evaluate_recall_at_k(recommended_ids, gt_items, k=1):
    hits = 0
    total = 0
    for rec_id, gts in zip(recommended_ids, gt_items):
        for gt in gts:
            if rec_id is not None and rec_id == gt:
                hits += 1
            total += 1
    recall = hits / total if total > 0 else 0.0
    return recall

def check_validity(file_path, model_key):
    """
    Checks whether the recommended item is in the candidate list.
    """
    candidates_key = f"candidates_{model_key}"
    recommended_key = f"recommended_{model_key}"

    total = 0
    valid = 0
    invalid_entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
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

def calculate_image_missing_proportion(data, item_meta, image_dir, candidate_type):
    total_candidates = 0
    missing_images = 0
    for entry in data:
        candidates = entry.get(candidate_type, [])
        for cid in candidates:
            total_candidates += 1
            image_path = os.path.join(image_dir, f"{cid}_0.jpg")
            if not os.path.exists(image_path):
                missing_images += 1

    proportion = (missing_images / total_candidates * 100) if total_candidates > 0 else 0
    print(f"Total candidates: {total_candidates}")
    print(f"Missing images: {missing_images}")
    print(f"Proportion of candidates without images: {proportion:.2f}%")

def prepare_candidate_info(candidates, item_meta, image_dir, default_image):
    candidate_info = []
    for cid in candidates:
        title = item_meta.get(cid, {}).get('title', 'No Title')
        image_path = os.path.join(image_dir, f"{cid}_0.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = default_image
        candidate_info.append({
            'id': cid,
            'title': title,
            'image': image
        })
    return candidate_info

def build_prompt(conversation_text, candidates_info):
    """
    Always uses images in the prompt.
    """
    prompt = (
        "You are an AI assistant specialized in providing personalized product recommendations based on user conversations. "
        "You are given a conversation between a user seeking recommendation (denoted by <submission>) and other users providing comments (denoted by <comment>). "
        "You are also given a set of candidate products with their IDs, titles and images formatted as \"ID: title\" followed by an image. "
        "Among the candidates, recommend the most relevant product to the seeker. "
        "Only reply with its ID, and don't say anything else.\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Candidates:\n"
    )
    for candidate in candidates_info:
        cid = candidate['id']
        title = candidate['title']
        prompt += f"{cid}: {title}\n"
        prompt += "".join(IMAGE_TOKENS) + "\n"

    prompt += "\nAssistant:"
    return prompt


##############################################################
# Dataset for Recommendation
##############################################################
class RecommendationDataset(Dataset):
    """
    Builds training or evaluation samples from conversation data,
    always using images in the prompt.
    """
    def __init__(self, data, item_meta, image_dir, candidate_type, default_image, is_training=True):
        self.data = data
        self.item_meta = item_meta
        self.image_dir = image_dir
        self.candidate_type = candidate_type
        self.default_image = default_image
        self.is_training = is_training

        if self.is_training:
            self.index_mapping = [
                (entry_idx, gt_idx)
                for entry_idx, entry in enumerate(self.data)
                for gt_idx in range(len(entry.get('gt_items', [])))
            ]
        else:
            self.index_mapping = [(entry_idx, None) for entry_idx in range(len(self.data))]

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        entry_idx, gt_idx = self.index_mapping[idx]
        entry = self.data[entry_idx]
        conversation_text = entry.get('context', '')
        gt_items = entry.get('gt_items', [])
        candidates = entry.get(self.candidate_type, [])

        if self.is_training:
            gt_item = gt_items[gt_idx]
            # Ensure exactly one ground truth in the candidate list
            if gt_item not in candidates and len(candidates) > 0:
                replace_idx = random.randint(0, len(candidates) - 1)
                candidates[replace_idx] = gt_item
            target_text = gt_item
        else:
            target_text = ""

        candidate_info = prepare_candidate_info(candidates, self.item_meta, self.image_dir, self.default_image)
        prompt = build_prompt(conversation_text, candidate_info)
        images = [c['image'] for c in candidate_info]

        return {
            'prompt': prompt,
            'images': images,
            'target_text': target_text,
            'gt_items': gt_items,
            'entry_idx': entry_idx
        }


##############################################################
# Data Collator for LLaVA v1.6
##############################################################
class DataCollatorForLLaVA:
    """
    Prepares batch inputs for LLaVA, with multi-subimage approach (5 per candidate).
    """
    def __init__(self, processor, tokenizer, max_length):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token_ids = [
            self.tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS
        ]

    def __call__(self, batch):
        prompts = [item['prompt'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        images_per_sample = [item['images'] for item in batch]

        full_prompts = [p + t for p, t in zip(prompts, target_texts)]

        # Tokenize
        tokenized_prompts = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        tokenized_full_prompts = self.tokenizer(
            full_prompts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = tokenized_full_prompts['input_ids']
        attention_mask = tokenized_full_prompts['attention_mask']
        labels = input_ids.clone()

        # Mask out the prompt part
        prompt_lengths = [len(ids) for ids in tokenized_prompts['input_ids']]
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mark image token positions
        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            for tid in self.image_token_ids:
                positions = (input_ids[b_idx] == tid).nonzero(as_tuple=False).squeeze(-1)
                image_token_mask[b_idx, positions] = True

        # Flatten images
        all_images = []
        for imgs in images_per_sample:
            all_images.extend(imgs)

        if all_images:
            images_processed = self.processor.image_processor(all_images, return_tensors='pt')
            images_tensor = images_processed['pixel_values']
        else:
            images_tensor = None

        images_per_sample_lengths = []
        for imgs in images_per_sample:
            subcount = 5 * len(imgs)  # 5 subimages per candidate
            images_per_sample_lengths.append(subcount)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images_tensor,
            'image_token_mask': image_token_mask,
            'images_per_sample_lengths': images_per_sample_lengths
        }


##############################################################
# LLaVAModel => forward => always uses images
##############################################################
class LLaVAModel(pl.LightningModule):
    """
    Model for recommendation prompt tuning:
    - Applies LoRA to the language model side.
    - Always uses images in the prompt.
    """
    def __init__(self, model, processor, tokenizer, args):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.args = args
        self.save_hyperparameters(ignore=['model', 'processor', 'tokenizer'])

        self.data_collator = DataCollatorForLLaVA(processor, tokenizer, max_length=args.max_length)
        self.test_results = []

    def forward(self, input_ids, attention_mask, images, image_token_mask, images_per_sample_lengths, labels=None):
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_token_mask = image_token_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Text embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        if images is not None:
            images = images.to(device, dtype=torch.float16)
            B_text = input_ids.size(0)
            B_prime = images.shape[0]
            num_views = images.shape[1]  # 5
            C, H, W = images.shape[2], images.shape[3], images.shape[4]

            # Flatten for vision tower
            images_reshaped = images.view(B_prime * num_views, C, H, W)
            with torch.no_grad():
                vision_outputs = self.model.vision_tower(images_reshaped, return_dict=False)
            cls_states = vision_outputs[0][:, 0, :]
            cls_states = cls_states.view(B_prime, num_views, -1)

            candidate_count = B_prime // B_text
            cls_states = cls_states.view(B_text, candidate_count, num_views, -1)

            cls_states = self.model.multi_modal_projector(cls_states)

            # Insert [CLS] states
            for b_idx in range(B_text):
                image_positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                if image_positions.numel() == 0:
                    continue

                needed_tokens = candidate_count * num_views
                pos_count = min(len(image_positions), needed_tokens)

                for c in range(candidate_count):
                    offset_c = c * num_views
                    for i in range(num_views):
                        idx_token = offset_c + i
                        if idx_token >= pos_count:
                            break
                        col = image_positions[idx_token].item()
                        inputs_embeds[b_idx, col, :] = cls_states[b_idx, c, i, :]

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'],
            image_token_mask=inputs['image_token_mask'],
            images_per_sample_lengths=inputs['images_per_sample_lengths'],
            labels=inputs['labels'],
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'],
            image_token_mask=inputs['image_token_mask'],
            images_per_sample_lengths=inputs['images_per_sample_lengths'],
            labels=inputs['labels'],
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs = self.data_collator(batch)
            device = next(self.model.parameters()).device

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            image_token_mask = inputs['image_token_mask'].to(device)
            images = inputs['images']
            images_per_sample_lengths = inputs['images_per_sample_lengths']

            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

            if images is not None:
                images = images.to(device, dtype=torch.float16)
                B_text = input_ids.size(0)
                B_prime = images.shape[0]
                num_views = images.shape[1]
                C, H, W = images.shape[2], images.shape[3], images.shape[4]

                images_reshaped = images.view(B_prime * num_views, C, H, W)
                with torch.no_grad():
                    vision_outputs = self.model.vision_tower(images_reshaped, return_dict=False)
                cls_states = vision_outputs[0][:, 0, :]
                cls_states = cls_states.view(B_prime, num_views, -1)

                candidate_count = B_prime // B_text
                cls_states = cls_states.view(B_text, candidate_count, num_views, -1)
                cls_states = self.model.multi_modal_projector(cls_states)

                for b_idx in range(B_text):
                    image_positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                    if image_positions.numel() == 0:
                        continue

                    needed_tokens = candidate_count * num_views
                    pos_count = min(len(image_positions), needed_tokens)
                    for c in range(candidate_count):
                        offset_c = c * num_views
                        for i in range(num_views):
                            idx_token = offset_c + i
                            if idx_token >= pos_count:
                                break
                            col = image_positions[idx_token].item()
                            inputs_embeds[b_idx, col, :] = cls_states[b_idx, c, i, :]

            generated_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            recommended_ids = []
            for txt in generated_texts:
                match = re.findall(r'\bB[A-Z0-9]{9}\b', txt.strip())
                recommended_ids.append(match[0][:10] if match else None)

            # gather ground truth
            gt_items_list = [b['gt_items'] for b in batch]
            entry_idxs = [b['entry_idx'] for b in batch]
            for i in range(len(batch)):
                self.test_results.append({
                    'entry_idx': entry_idxs[i],
                    'recommended_id': recommended_ids[i],
                    'response': generated_texts[i]
                })

            recall = evaluate_recall_at_k(recommended_ids, gt_items_list, k=1)
            self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
            return {'test_recall': recall}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        return optimizer


##############################################################
# Re-introduce LoRA on the Language Model
##############################################################
def find_llm_linear_layer_names(llm_module, prefix="language_model"):
    """
    Recursively find all nn.Linear layers in the language model part.
    """
    import torch.nn as nn
    linear_names = []
    for name, module in llm_module.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            linear_names.append(full_name)
    return linear_names


def main():
    parser = argparse.ArgumentParser(description="Recommendation Prompt Tuning with pretrained LLaVA v1.6 + LM LoRA.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the base or partially LoRA-updated model directory.")
    parser.add_argument("--candidate_type", type=str, default="candidates_st",
                        help="JSON key for candidate items.")
    parser.add_argument("--finetune_output_dir", type=str, default="./out_finetuned",
                        help="Directory to save finetuning results.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Required dataset info
    parser.add_argument("--item_meta_path", type=str, default="../data/item2meta_train.json")
    parser.add_argument("--image_dir", type=str, default="../data/train_images")
    parser.add_argument("--category", type=str, default='all_beauty')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the base or partially LoRA-updated model
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16
    ).to(device)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    processor.tokenizer.padding_side = "right"
    tokenizer = processor.tokenizer

    # 2) Add 5 special tokens
    special_tokens_dict = {'additional_special_tokens': IMAGE_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))

    # 3) LoRA to Language Model (re-introduce)
    lm_linear_names = find_llm_linear_layer_names(base_model.language_model, prefix="language_model")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=lm_linear_names
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    # 4) Load metadata & data
    args.train_data_path = f"../data/{args.category}/train.jsonl"
    args.val_data_path = f"../data/{args.category}/valid.jsonl"
    args.test_data_path = f"../data/{args.category}/test.jsonl"
    item_meta = load_item_meta(args.item_meta_path)
    train_data = load_jsonl(args.train_data_path)
    val_data = load_jsonl(args.val_data_path)
    test_data = load_jsonl(args.test_data_path)

    print("Calculating missing images (train):")
    calculate_image_missing_proportion(train_data, item_meta, args.image_dir, args.candidate_type)
    print("Calculating missing images (valid):")
    calculate_image_missing_proportion(val_data, item_meta, args.image_dir, args.candidate_type)
    print("Calculating missing images (test):")
    calculate_image_missing_proportion(test_data, item_meta, args.image_dir, args.candidate_type)

    default_image = Image.new('RGB', (336, 336), color=(255, 255, 255))

    train_dataset = RecommendationDataset(
        data=train_data,
        item_meta=item_meta,
        image_dir=args.image_dir,
        candidate_type=args.candidate_type,
        default_image=default_image,
        is_training=True
    )
    val_dataset = RecommendationDataset(
        data=val_data,
        item_meta=item_meta,
        image_dir=args.image_dir,
        candidate_type=args.candidate_type,
        default_image=default_image,
        is_training=False
    )
    test_dataset = RecommendationDataset(
        data=test_data,
        item_meta=item_meta,
        image_dir=args.image_dir,
        candidate_type=args.candidate_type,
        default_image=default_image,
        is_training=False
    )

    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    llava_model = LLaVAModel(model, processor, tokenizer, args)
    os.makedirs(args.finetune_output_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        callbacks=[],
        precision='16',
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # Train
    print("[INFO] Starting training for recommendation tuning (LM LoRA).")
    trainer.fit(llava_model, train_loader, val_loader)

    # Test
    print("[INFO] Testing on test data.")
    test_results = trainer.test(llava_model, dataloaders=test_loader)
    print(f"Test Results: {test_results}")

    test_results_list = llava_model.test_results
    results_by_idx = {res['entry_idx']: res for res in test_results_list}

    recommended_ids = []
    ground_truths = []
    for idx, entry in enumerate(test_data):
        result = results_by_idx.get(idx)
        if result:
            recommended_ids.append(result['recommended_id'])
            ground_truths.append(entry.get('gt_items', []))
        else:
            recommended_ids.append(None)
            ground_truths.append(entry.get('gt_items', []))

    recall = evaluate_recall_at_k(recommended_ids, ground_truths, k=1)
    print(f"[Test] Recall@1: {recall:.4f}")

    # Prepare recommended/response fields
    model_key = "st" if args.candidate_type == 'candidates_st' else "gpt_large"
    if model_key == "st":
        recommended_field = 'recommended_st'
        response_field = 'response_st'
    else:
        recommended_field = f"recommended_{model_key}"
        response_field = f"response_{model_key}"

    # Save final test outputs
    for idx, entry in enumerate(test_data):
        res = results_by_idx.get(idx)
        if res:
            entry[recommended_field] = res['recommended_id']
            entry[response_field] = res['response']

    out_file_name = f'test_results_{args.candidate_type}.jsonl'
    output_file_path = os.path.join(args.finetune_output_dir, out_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in test_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print(f"Test details saved to {output_file_path}")

    validity, invalid_entries, total_count = check_validity(output_file_path, model_key)
    print(f"Validity@1: {validity:.4f}, invalid entries: {len(invalid_entries)} / {total_count}")

    # Save final LoRA adapter
    trained_lora_path = os.path.join(args.finetune_output_dir, 'trained_lora_adapter')
    model.save_pretrained(trained_lora_path)
    print(f"LoRA adapter saved to {trained_lora_path}")

    # Summaries
    summary_file = os.path.join(args.finetune_output_dir, "results_summary.csv")
    csv_exists = os.path.exists(summary_file)
    with open(summary_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["model_dir", "candidate_type", "lr", "weight_decay",
                             "num_epochs", "recall@1", "validity@1", "output_file"])
        writer.writerow([
            args.model_dir,
            args.candidate_type,
            args.lr,
            args.weight_decay,
            args.num_epochs,
            recall,
            validity,
            out_file_name
        ])
    print(f"Results summary updated: {summary_file}")
    print("Done finetuning (LM LoRA) and testing with images.")


if __name__ == "__main__":
    main()
