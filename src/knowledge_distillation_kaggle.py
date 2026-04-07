#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC: Visual Knowledge Self-Distillation — Kaggle T4 Adaptation
- 4-bit QLoRA (NF4) on frozen LLM
- device_map={"": 0} single GPU
- Calls full PEFT model wrapper for loss (same as original)
"""

import argparse
import json
import math
import os

import pytorch_lightning as pl
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]

PROMPT_TEMPLATE = (
    "You are a helpful assistant.\n"
    "Given an Amazon product's title and its image, please provide a detailed, visually grounded description of the product "
    "that would help someone decide whether to purchase it. "
    "Focus on the product's appearance, features, and any other visually informative aspects. "
    "Do not mention the product's title in your answer. "
    "This product's title is: {title}\n"
    f"{''.join(IMAGE_TOKENS)}\n\n"
    "Assistant:"
)


class ImageDescriptionDataset(Dataset):
    def __init__(self, data_source, images_dir, is_training=True, default_image_size=(336, 336)):
        super().__init__()
        self.images_dir = images_dir
        self.default_image = Image.new('RGB', default_image_size, (255, 255, 255))
        self.data = []
        if data_source.endswith('.json'):
            with open(data_source, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            for asin, item_data in data_json.items():
                title = item_data.get("title", "No Title")
                for image_name, desc in item_data.get("image_descriptions_llava_cleaned", {}).items():
                    image_path = os.path.join(images_dir, image_name)
                    if os.path.exists(image_path):
                        self.data.append({"title": title, "image_path": image_path, "description": desc})
        elif data_source.endswith('.jsonl'):
            with open(data_source, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    image_path = os.path.join(images_dir, entry.get("image_name", ""))
                    if os.path.exists(image_path):
                        self.data.append({"title": entry.get("title", ""), "image_path": image_path, "description": entry.get("image_description_llava_cleaned", "")})

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB") if os.path.exists(item["image_path"]) else self.default_image
        return {"title": item["title"], "image": image, "description": item["description"]}


class DataCollator:
    def __init__(self, processor, tokenizer, max_length, prompt_template):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.image_token_ids = [tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS]

    def __call__(self, batch):
        prompts = [self.prompt_template.format(title=item["title"]) for item in batch]
        target_texts = [item["description"] for item in batch]
        images = [item["image"] for item in batch]
        full_prompts = [p + t for p, t in zip(prompts, target_texts)]

        tokenized_prompts = self.tokenizer(prompts, max_length=self.max_length, padding='longest', truncation=True, return_tensors='pt')
        tokenized_full = self.tokenizer(full_prompts, max_length=self.max_length, padding='longest', truncation=True, return_tensors='pt')

        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']
        labels = input_ids.clone()
        for i, plen in enumerate([len(x) for x in tokenized_prompts['input_ids']]):
            labels[i, :plen] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            for tk_id in self.image_token_ids:
                positions = (input_ids[b_idx] == tk_id).nonzero(as_tuple=True)
                image_token_mask[b_idx, positions] = True

        images_tensor = self.processor.image_processor(images, return_tensors='pt')['pixel_values']
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'images': images_tensor, 'image_token_mask': image_token_mask}


def get_llava_core(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for cand in candidates:
        if cand and hasattr(cand, "vision_tower") and hasattr(cand, "multi_modal_projector"):
            return cand
    raise AttributeError("Could not find LLaVA core with vision_tower and multi_modal_projector")


class PretrainVisionModel(pl.LightningModule):
    def __init__(self, model, processor, tokenizer, args):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.args = args
        self.data_collator = DataCollator(processor, tokenizer, max_length=args.max_length, prompt_template=PROMPT_TEMPLATE)
        self.save_hyperparameters(ignore=['model', 'processor', 'tokenizer'])
        self.val_loss_sum = 0.0
        self.val_token_count = 0

    def forward(self, input_ids, attention_mask, images, image_token_mask, labels=None):
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
            B, num_views, C, H, W = images.shape
            vision_outputs = core.vision_tower(images.view(B * num_views, C, H, W))
            cls_states = vision_outputs.last_hidden_state[:, 0, :].view(B, num_views, -1)
            cls_states = core.multi_modal_projector(cls_states)
            for b_idx in range(B):
                positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                for i in range(min(len(positions), num_views)):
                    inputs_embeds[b_idx, positions[i].item(), :] = cls_states[b_idx, i, :]

        # Call full model wrapper — handles loss automatically
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(**{k: inputs[k] for k in ['input_ids', 'attention_mask', 'images', 'image_token_mask']}, labels=inputs['labels'])
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(**{k: inputs[k] for k in ['input_ids', 'attention_mask', 'images', 'image_token_mask']}, labels=inputs['labels'])
        val_loss = outputs.loss
        num_tokens = (inputs['labels'] != -100).sum().item()
        self.val_loss_sum += val_loss.item() * num_tokens
        self.val_token_count += num_tokens
        return val_loss

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_sum / self.val_token_count if self.val_token_count > 0 else float('inf')
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_perplexity', ppl, prog_bar=True)
        if self.global_rank == 0:
            with open(os.path.join(self.args.output_dir, f"val_metrics_epoch_{self.current_epoch+1}.txt"), "w") as f:
                f.write(f"Val Loss: {avg_val_loss}\nVal PPL: {ppl}\n")
        self.val_loss_sum = 0.0
        self.val_token_count = 0

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.args.lr, weight_decay=self.args.weight_decay)


def find_vision_linear_layer_names(vision_model, prefix="vision_tower"):
    return [f"{prefix}.{name}" if prefix else name for name, module in vision_model.named_modules() if isinstance(module, torch.nn.Linear)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--train_data", type=str, default="/kaggle/working/ds8008-group8-lavic/data/item2meta_train.json")
    parser.add_argument("--val_data", type=str, default="/kaggle/working/ds8008-group8-lavic/data/item2meta_valid.jsonl")
    parser.add_argument("--train_images_dir", type=str, default="/kaggle/working/ds8008-group8-lavic/data/train_images")
    parser.add_argument("--val_images_dir", type=str, default="/kaggle/working/ds8008-group8-lavic/data/valid_images")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/out_distilled")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate_before_training", action="store_true")
    args = parser.parse_args([])

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': IMAGE_TOKENS})
    tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))

    core = get_llava_core(base_model)
    target_modules = (
        find_vision_linear_layer_names(core.vision_tower, prefix="vision_tower") +
        find_vision_linear_layer_names(core.multi_modal_projector, prefix="multi_modal_projector")
    )
    print(f"[INFO] Applying LoRA to {len(target_modules)} vision layers")

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    pl_model = PretrainVisionModel(lora_model, processor, tokenizer, args)

    train_dataset = ImageDescriptionDataset(args.train_data, args.train_images_dir, is_training=True)
    val_dataset = ImageDescriptionDataset(args.val_data, args.val_images_dir, is_training=False)
    print(f"[INFO] Train dataset size: {len(train_dataset)}")
    print(f"[INFO] Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: x)

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='pretrain_epoch{epoch}-val_loss{val_loss:.4f}', save_top_k=1, monitor='val_loss', mode='min')

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
    )

    print("\n[INFO] Starting vision distillation training.")
    trainer.fit(pl_model, train_loader, val_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"[INFO] Best checkpoint: {best_ckpt_path}")
    best_model = PretrainVisionModel.load_from_checkpoint(best_ckpt_path, model=lora_model, processor=processor, tokenizer=tokenizer, args=args)
    best_model.model.save_pretrained(os.path.join(args.output_dir, "vision_lora_adapter_best"))
    print("[INFO] Best LoRA adapter saved to", os.path.join(args.output_dir, "vision_lora_adapter_best"))


if __name__ == "__main__":
    main()
