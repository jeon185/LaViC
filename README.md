# Reimplementation of LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation

[![DOI](https://zenodo.org/badge/930728835.svg)](https://doi.org/10.5281/zenodo.15560047)

> **DS8008 NLP (Text Mining) — Final Project**
> Toronto Metropolitan University
> Professor: Syed Shariyar Murtaza
> Due: April 18, 2026

---

## Group 8

| Name | Student ID |
|---|---|
| Jason Yu | 501048589 |
| Jessie Ma | 501274167 |
| Yosef Moustafa | 501390640 |

---

## Project Overview

This repository contains Group 8's reimplementation of **LaViC** (Large Vision-Language Conversational Recommendation Framework), originally proposed by Jeon et al. and published at **KDD 2025**.

LaViC addresses a core challenge in visually-aware conversational recommendation: integrating product images into dialogue-based recommender systems without incurring the prohibitive computational cost of processing hundreds of image tokens per item (the *token explosion* problem).

The framework operates in two stages:
1. **Visual Knowledge Self-Distillation** — compresses each product image from thousands of patch tokens down to just 5 [CLS]-positioned embeddings using LoRA fine-tuning of the vision module.
2. **Recommendation Prompt Tuning** — fine-tunes the large language model to select the correct item from candidates given the conversation context and compressed image embeddings.

We also analyze LaViC in relation to **Rec-GPT4V**, a zero-shot multimodal baseline that uses GPT-4Vision for item ranking, to highlight the efficiency and accuracy trade-offs between the two approaches.

- **Original Paper:** [arXiv:2503.23312](https://arxiv.org/abs/2503.23312)
- **Original Repository:** [github.com/jeon185/LaViC](https://github.com/jeon185/LaViC)

---

## Domain Focus

For this reimplementation, we focus on the **Home** domain (`amazon_home`) from the Reddit-Amazon dataset.

---

## Repository Structure
```plaintext
ds8008-group8-lavic/
  ├── notebooks/
  │   └── LaViC_Report.ipynb
  ├── presentation/
  │   └── group8_slides.pdf
  ├── data/
  │   ├── amazon_home/
  │   │   ├── train.jsonl
  │   │   ├── valid.jsonl
  │   │   └── test.jsonl
  │   ├── train_images/
  │   ├── valid_images/
  │   ├── item2meta_train.json
  │   └── item2meta_valid.jsonl
  └── src/
      ├── crawl_images.py
      ├── knowledge_distillation.py
      └── prompt_tuning.py
```

---

## Work Flow

### 1. Environment Setup
```bash
cd ds8008-group8-lavic
pip install -r requirements.txt
```

### 2. Image Crawling and Subsetting
This will take around 2 hours
```bash
cd src
python crawl_images.py
```



### 3. Visual Knowledge Self-Distillation
```bash
python knowledge_distillation.py \
  --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  --train_data ../data/item2meta_train.json \
  --val_data ../data/item2meta_valid.jsonl \
  --train_images_dir ../data/train_images \
  --val_images_dir ../data/valid_images \
  --output_dir ./out_distilled \
  --lr 5e-5 --weight_decay 1e-5 --num_epochs 2 --batch_size 4
```

### 4. Recommendation Prompt Tuning
```bash
python prompt_tuning.py \
  --model_dir ./out_distilled/vision_lora_adapter_best \
  --candidate_type candidates_st \
  --finetune_output_dir ./out_finetuned \
  --max_length 2048 \
  --batch_size 1 \
  --lr 5e-5 --weight_decay 1e-5 \
  --num_epochs 1 \
  --item_meta_path ../data/item2meta_train.json \
  --image_dir ../data/train_images \
  --category amazon_home
```

---

## Citation

This project is based on the following work:
```bibtex
@inproceedings{jeon25adapting,
  title     = "Adapting large vision-language models to visually-aware conversational recommendation",
  author    = "Hyunsik Jeon and Satoshi Koide and Yu Wang and Zhankui He and Julian McAuley",
  year      = "2025",
  booktitle = "KDD"
}
```

---

*This reimplementation is submitted as a final project for DS8008 NLP (Text Mining) at Toronto Metropolitan University and is intended for academic purposes only.*
