# 🩺 Deep Learning for Dermatology: Automated Skin Lesion Classification (HAM10000)

This repository presents a **comparative study and implementation of deep learning architectures** for the automated classification of 7 dermatoscopic skin lesion types. Beyond standard CNNs, this project addresses **real-world medical imaging challenges** including data leakage, severe class imbalance, and the need for interpretability in clinical settings.

---

##  Project Mission

Medical AI is **high-stakes**: false negatives in melanoma detection can have life-threatening consequences. This project goes beyond raw accuracy by:

- Prioritizing **rare and high-risk classes** like Melanoma (mel).  
- Ensuring **patient-level generalization** through leakage-free splitting.  
- Integrating **explainability tools** to build clinical trust.  

**Objective:** Classify dermatoscopic images into **7 diagnostic categories** with a focus on both predictive performance and interpretability.

---

##  Engineering & Clinical Challenges

### 1. Patient-Level Data Leakage
**Problem:** HAM10000 contains multiple images of the same lesion. Random splits allow models to memorize lesions rather than learn disease patterns.  

**Solution:** Custom **lesion_id-based splitting**, ensuring no patient appears in both training and test sets.

---

### 2. Extreme Class Imbalance (58:1)
**Problem:** 6,705 Melanocytic nevi (nv) vs. 115 Dermatofibroma (df) samples. A naive model would predict majority classes almost exclusively.  

**Solution:**  
- Weighted Cross-Entropy Loss based on **inverse class frequency**.  
- Targeted data augmentation for minority classes.  
- Focused evaluation metrics: **Macro F1-score, Class Recall**, especially for high-risk lesions.

---

### 3. The Clinical "Black Box" Problem
**Problem:** Clinicians require visual evidence of model reasoning to trust predictions.  

**Solution:** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps, highlighting **textures and pigment networks** that guide the CNN’s decisions.

---

##  Model Architectures & Training Strategy

### Baseline CNN
- 4-block custom CNN trained from scratch.  
- Achieved **~70% accuracy**, serving as a difficulty floor.

### Transfer Learning Models
Two-phase strategy implemented for **ResNet50** and **EfficientNet-B0**:

**Phase 1: Feature Extraction (Warm-up)**  
- Freeze ImageNet-pretrained backbone.  
- Train custom classification head only.  
- Goal: Stabilize newly added dense layers without disrupting pretrained spatial features.

**Phase 2: Fine-Tuning (Progressive Unfreezing)**  
- Unfreeze entire network.  
- Apply **differential learning rates** (10x smaller for backbone).  
- Goal: Adapt low-level filters to dermatoscopic textures and patterns.

---

##  Performance Overview

| Model             | Overall Accuracy | Macro Avg F1 | Weighted Avg F1 |
|------------------|----------------|--------------|----------------|
| Baseline CNN      | ~70.1%         | 0.32         | 0.68           |
| ResNet50          | 73.27%         | 0.43         | 0.73           |
| EfficientNet-B0   | 78.41%         | 0.54         | 0.78           |

**Key Insights:**  
- **"NV" Dominance:** Both models excel on Melanocytic nevi (nv) due to large support (F1 ~0.89).  
- **Architecture Edge:** EfficientNet-B0 outperforms ResNet50 on minority classes like Actinic keratoses (akiec), F1 = 0.55 vs. 0.26.  
- **Data Floor:** Dermatofibroma (df) remains challenging (0% recall), highlighting a limitation of convolutional architectures on extremely rare lesions.

---

## 🩺 Per-Class Performance Comparison

| Class | ResNet50 F1 | EfficientNet-B0 F1 | ResNet50 Recall | EfficientNet-B0 Recall |
|-------|-------------|-------------------|----------------|-----------------------|
| akiec  | 0.27        | 0.55              | 0.20           | 0.54                  |
| bcc    | 0.42        | 0.67              | 0.37           | 0.69                  |
| bkl    | 0.51        | 0.56              | 0.66           | 0.60                  |
| df     | 0.00        | 0.00              | 0.00           | 0.00                  |
| mel    | 0.39        | 0.41              | 0.38           | 0.35                  |
| nv     | 0.87        | 0.89              | 0.85           | 0.91                  |
| vasc   | 0.55        | 0.72              | 0.52           | 0.62                  |

> EfficientNet-B0 consistently improves **minority class recognition**, demonstrating the importance of architecture selection in medical datasets.

---

##  Technical Stack

- **Framework:** PyTorch 2.0+ (MPS acceleration for Apple Silicon)  
- **Computer Vision:** Torchvision, PIL, OpenCV  
- **Data Processing & Analysis:** NumPy, Pandas, Scikit-learn (weighted metrics)  
- **Visualization:** Matplotlib, Seaborn, Grad-CAM  

---

##  Project Structure

```plaintext
├── models/                        # Saved model checkpoints (.pth)
├── README.md                       # Project overview and documentation
├── data/                           # HAM10000 images & metadata.csv
│   ├── 01_data_exploration.ipynb   # Data exploration and lesion-ID splitting
│   ├── 02_data_processing.ipynb   # CNN, ResNet50, EfficientNet-B0 architectures
│   ├── 03_baseline_model.ipynb     # Baseline model training & Grad-CAM visualizations
│   └── 04_advanced_models.ipynb    # Advanced model experiments and fine-tuning
└── requirements.txt 
