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
Place both create_home_sub_images.py and make_home_item2meta_subset.py inside the **data** folder, then call CMD inside the data folder

**Before you run create_home_sub_images.py**
 - The code will extract exactly 2 images per item by default, that is, a training image size of 5674 images
 - You can change this by editing the .py with **MAX_IMAGES_PER_ITEM = 2 on line 11**

Run these two lines in order:
```bash
python create_home_sub_images.py
python make_home_item2meta_subset.py
```
Two things will be created
 - amazon_home_train_images_subset (Folder)
 - item2meta_train_amazon_home.json

### 3. Upload to Google Colab
You will upload these items to your Google Drive or Colab environment in this order
```plaintext
├── LaViC/
|    ├── data/
|    │   ├── amazon_home_train_images_subset (Image Folder)
|    │   ├── amazon_home
|    │   │   ├── train.jsonl
|    │   │   ├── valid.jsonl
|    │   │   └── test.jsonl
|    │   ├── valid_images (Image Folder)
|    │   ├── item2meta_train_amazon_home.json
|    │   └── item2meta_valid.jsonl
|    └── src/
|        ├── knowledge_distillation.py
|        └── prompt_tuning.py
├──LaViC.ipynb
```
### 4. Run prechecks and Notes
 - Run Cell 1 to mount your Google Drive, if you have uploaded all the files to it
 - Run Cell 2-4 to check GPU and packages
 - Note: We used A100 GPU for this project, and it is not included in free version of Colab, you may run into disk space issues in the middle, depending on your disk useage.

### 5. Visual Knowledge Self-Distillation & Recommendation Prompt Tuning
 - Run Cell 5 for Visual Knowledge Self-Distillation, and at the end it will create two files
 - Run Cell 6 for Recommendation Prompt Tuning and testing, this .py file fine-tunes the model, then run the test on the final model.  Result will be shown on screen at the end
 - Note: expect to spend from 25-40 minutes per Cell block here, with A100 GPU.  Around 26 hours per cell with T4 GPU


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
