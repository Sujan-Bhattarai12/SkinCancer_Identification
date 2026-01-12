# Skin Lesion Classification using Deep Learning

## 🎯 Problem Statement

Skin cancer is one of the most common types of cancer worldwide, with melanoma being the deadliest form. Early detection is crucial for successful treatment, but accurate diagnosis requires expertise that may not be readily available in all regions. This project aims to develop an automated deep learning system to classify skin lesions into 7 different categories using dermatoscopic images from the HAM10000 dataset, potentially assisting dermatologists in early detection and diagnosis.

**Classes:**
- Melanoma (mel) - Most dangerous
- Melanocytic nevi (nv)
- Basal cell carcinoma (bcc)
- Actinic keratoses (akiec)
- Benign keratosis (bkl)
- Dermatofibroma (df)
- Vascular lesions (vasc)

## 🔬 Method Used

### 1. Data Preprocessing
- **Dataset:** 10,015 dermatoscopic images from HAM10000
- **Split Strategy:** 70/15/15 (train/val/test) by `lesion_id` to prevent data leakage from duplicate lesions
- **Data Augmentation:** Random flips, rotation, color jitter, and affine transformations
- **Normalization:** ImageNet statistics for transfer learning compatibility

### 2. Models Implemented

**Baseline Model:**
- Custom CNN with 4 convolutional blocks (~5M parameters)
- Trained from scratch as performance benchmark

**Transfer Learning Models:**
- **ResNet50:** Pre-trained on ImageNet with custom classifier
- **EfficientNet-B0:** Efficient architecture with fewer parameters
- **Training Strategy:** Two-phase approach
  - Phase 1: Freeze backbone, train classifier only (10 epochs)
  - Phase 2: Unfreeze last layers, fine-tune entire model (15 epochs)

### 3. Techniques for Class Imbalance
- Weighted Cross-Entropy Loss (inverse class frequency)
- Extensive data augmentation
- Stratified splitting

## 4. Model Interpretability
- **Grad-CAM visualizations** to understand model decisions
- Heatmaps showing which image regions influence predictions#

## 📊 Model Output

| Model | Validation Accuracy | Test Accuracy | Parameters |
|-------|-------------------|---------------|------------|
| Baseline CNN | ~70% | ~68% | 5M |
| **ResNet50** | **~85%** | **~83%** | 24.5M |
| EfficientNet-B0 | ~84% | ~82% | 4.8M |

**Key Metrics (ResNet50):**
- **Micro-average AUC:** 0.95+
- **Melanoma Detection:** Precision: ~0.80, Recall: ~0.75
- **Best Performing Class:** Melanocytic nevi (F1: 0.90+)
- **Most Challenging:** Dermatofibroma (limited samples)

**Evaluation Includes:**
- Confusion matrices
- ROC curves with AUC scores for all classes
- Per-class precision, recall, F1-scores
- Confidence analysis
- Error pattern analysis with Grad-CAM

## 🚧 Challenging Parts

### 1. **Severe Class Imbalance**
- **Problem:** Melanocytic nevi (nv) had 6,705 samples while Dermatofibroma (df) had only 115 samples (~58:1 ratio)
- **Solution:** 
  - Implemented weighted loss function with inverse frequency weights
  - Applied aggressive data augmentation to minority classes
  - Used stratified splitting to maintain class distribution across splits

### 2. **Data Leakage Prevention**
- **Problem:** Same lesion photographed multiple times appears in dataset
- **Solution:** Split by `lesion_id` instead of `image_id` to ensure no lesion appears in both training and test sets
- **Impact:** Critical for real-world performance estimation

### 3. **Limited Computational Resources**
- **Problem:** Large models and dataset require significant GPU memory
- **Solution:**
  - Used efficient architectures (EfficientNet-B0)
  - Implemented two-phase training to reduce memory footprint
  - Batch size optimization (32) for GPU memory management
  - Transfer learning instead of training from scratch

### 5. **Model Interpretability**
- **Problem:** Black-box models unacceptable in medical applications
- **Solution:**
  - Implemented Grad-CAM visualizations
  - Analyzed misclassification patterns
  - Provided confidence scores with predictions
  - Created visual explanations showing regions of interest

## 🛠️ Tech Stack

- **Framework:** PyTorch 2.0+
- **Models:** ResNet50, EfficientNet-B0
- **Visualization:** Matplotlib, Seaborn, Grad-CAM
- **Metrics:** scikit-learn, ROC-AUC
- **Data Processing:** NumPy, Pandas, PIL, OpenCV

## 📁 Project Structure

```
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_model_evaluation_visualization.ipynb
├── data/
│   ├── ham10000/
│   └── split CSVs
├── models/
│   ├── resnet50_best.pth
│   └── efficientnet_b0_best.pth
├── results/
│   └── visualizations & reports
└── requirements.txt
```

## 💡 Key Takeaways

- Transfer learning provides 10-15% accuracy improvement over baseline
- Class imbalance is the biggest challenge in medical imaging
- Interpretability (Grad-CAM) is essential for clinical applications
- Proper data splitting prevents optimistic performance estimates
- Melanoma detection requires high recall to avoid missing cases

---

**Author:** [SUJAN BHTTARAI]  
**Dataset:** [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)  
**License:** MIT