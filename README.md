# Deep Learning for Dermatology: Automated Skin Lesion Classification
This repository presents a comparative study and implementation of deep learning architectures for the classification of 7 dermatoscopic skin lesion types. Beyond standard CNNs, this project addresses real-world medical imaging challenges including data leakage, severe class imbalance, and the need for interpretability in clinical settings.

##  Project Mission
Medical AI is **high-stakes**: false negatives in melanoma detection can have life-threatening consequences. This project goes beyond raw accuracy by:

- Prioritizing **rare and high-risk classes** like Melanoma (mel).  
- Ensuring **patient-level generalization** through leakage-free splitting.  

**Objective:** Classify dermatoscopic images into **7 diagnostic categories** with a focus on both predictive performance.

**Model**: Implemented ConvNeXt-Base architecture with two-phase training: 25 epochs of head-only training with frozen backbone, followed by 10 epochs of fine-tuning with the last 3 layers unfrozen. Engineered Focal Loss (α=1, γ=2) to mitigate class imbalance in the HAM10000 dataset. Applied data augmentation including random horizontal/vertical flips, ±20° rotation, and color jittering (brightness, contrast, saturation, hue ±0.1-0.2). Optimized using AdamW with 1e-2 weight decay and adaptive learning rate scheduling (1e-3 during phase 1, 1e-4 during phase 2). Deployed the model via Gradio web application for real-time inference on skin lesion images.

**Output**
| Class | Correct / Total | Accuracy (%) | Most Common Confusion |
|------:|-----------------|--------------|------------------------|
| 0 | 42 / 59 | 71.2 | Confused with 2 (11×, 19%) |
| 1 | 79 / 91 | 86.8 | Confused with 2 (5×, 5%) |
| 2 | 109 / 155 | 70.3 | Confused with 5 (18×, 12%) |
| 3 | 28 / 30 | 93.3 | Confused with 0 (1×, 3%) |
| 4 | 105 / 189 | 55.6 | Confused with 5 (54×, 29%) |
| 5 | 892 / 969 | 92.1 | Confused with 2 (27×, 3%) |
| 6 | 18 / 22 | 81.8 | Confused with 1 (1×, 5%) |


**Overall Accuracy:** 85.52% (1273 / 1505)
**Validation Accuracy:** 84.02% (1266 / 1502)