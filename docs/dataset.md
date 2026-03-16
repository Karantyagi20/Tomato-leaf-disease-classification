# 📁 Dataset Documentation

## Overview

The dataset used in this project is a structured collection of **tomato leaf images** organised into class-labelled directories. It covers **4 categories** — three disease classes and one healthy reference class — spanning a total of **763 images**.

---

## Directory Structure

Place your dataset in the project root under a folder named `dataset/`:

```
dataset/
├── Tomato_Early_blight/     # 144 images
├── Tomato_Late_blight/      # 180 images
├── Tomato_Leaf_Mold/        # 192 images
└── Tomato_healthy/          # 247 images
```

> ⚠️ The dataset folder is excluded from version control via `.gitignore`.  
> Download the images separately and place them in the structure above.

---

## Class Distribution

| Class | Images | Share (%) | Description |
|---|:---:|:---:|---|
| **Tomato Early Blight** | 144 | 18.9% | Caused by *Alternaria solani* — dark concentric ring lesions on older leaves |
| **Tomato Late Blight** | 180 | 23.6% | Caused by *Phytophthora infestans* — water-soaked lesions, white sporulation |
| **Tomato Leaf Mold** | 192 | 25.2% | Caused by *Passalora fulva* — yellow patches on upper surface, mold beneath |
| **Tomato Healthy** | 247 | 32.4% | Disease-free leaves — negative reference class |
| **Total** | **763** | **100%** | |

---

## Dataset Split

| Model | Train | Test | Total | Strategy |
|---|:---:|:---:|:---:|---|
| **Random Forest** | ~610 | 153 | 763 | Stratified 80/20 split |
| **K-Means** | — | — | 763 | No split (unsupervised) |

> Stratified splitting ensures the class distribution (18.9% / 23.6% / 25.2% / 32.4%) is preserved in both train and test subsets.

---

## Preprocessing Pipeline

Every image goes through the following transformations before entering any model:

| Step | Detail |
|---|---|
| **Read** | `cv2.imread()` — loads as BGR |
| **Resize** | `64 × 64` pixels |
| **Colour** | BGR → RGB (`cv2.COLOR_BGR2RGB`) |
| **Flatten** | 64 × 64 × 3 = **12,288 features** per image |
| **Encode** | String labels → integers via `sklearn.LabelEncoder` |
| **PCA** | 12,288D → **120 components** (81.37% variance retained) |

---

## Source

> The dataset is derived from the **PlantVillage** benchmark collection.  
> Reference: Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health.* arXiv:1511.08060.

---

## Notes

- Images vary in lighting, angle, and disease progression stage — providing realistic field conditions.
- The dataset exhibits **mild class imbalance** (Early Blight is underrepresented at 18.9%).
- No data augmentation was applied in this study — a potential improvement for future work.
