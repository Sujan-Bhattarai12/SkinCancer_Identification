# 🩺 Medical Computer Vision: Skin Lesion Classification (HAM10000)

## Project Mission
The goal of this project was **to train a model**, and tackle the unique architectural and statistical challenges of medical imaging.I built a system to classify dermatoscopic images into **7 diagnostic categories**, with a primary focus on **high-stakes detection of Melanoma**.

---

## Engineering Challenges

### 1. Data Leakage
The **HAM10000 dataset** contains multiple images of the same lesion. A naive random split would place different views of the same lesion in both training and validation sets, inflating the reported accuracy.  
**The Fix:**  
Implemented a **custom splitting logic based on `lesion_id`** to ensure that the model is tested on **unseen patients**, not just unseen images of known lesions.

---

### 2. Handling Severe Class Imbalance
The dataset is heavily dominated by **Melanocytic nevi (nv)**. Without intervention, a model could reach ~67% accuracy by always predicting "nv".  
**The Fix:**  
- Moved beyond simple oversampling.  
- Implemented **Weighted Cross-Entropy Loss**:  

\[
W_c = \frac{N}{C \cdot n_c}
\]  

where:
- \(N\) = total number of samples  
- \(C\) = total number of classes  
- \(n_c\) = number of samples in class \(c\)  

This ensures that gradients prioritize minority classes like **Dermatofibroma** and **Vascular lesions**.

---

### 3. Medical Interpretability (Explainability)
In healthcare, **predictions without explanations are not clinically useful**.  
**The Fix:**  
Integrated **Grad-CAM**, which produces heatmaps highlighting the areas of the image the CNN focuses on, such as **specific textures or pigment networks**, providing insight into model decisions.

---
## 🏗️ Technical Architecture

### Baseline: Custom CNN
- Built a **4-block CNN from scratch** to measure the difficulty floor.  
- Plateaued at ~70% accuracy, confirming that **skin textures require deeper hierarchical features**.

### The "Winner": ResNet50 + Transfer Learning
**Two-phase training strategy:**

1. **Phase 1:** Freeze ImageNet weights, train only the custom head for 10 epochs to stabilize randomly initialized weights.  
2. **Phase 2:** Unfreeze deeper layers and fine-tune with a **lower learning rate (1e-5)** to adapt filters to dermatoscopic patterns.
---

## Performance Benchmarks

| Metric                     | Baseline CNN | ResNet50 (Fine-tuned) |
|-----------------------------|--------------|----------------------|
| Test Accuracy               | 68.2%        | 83.4%                |
| Micro-Average AUC           | 0.82         | 0.96                 |
| Inference Latency           | ~15 ms       | ~28 ms               |
| Melanoma Recall             | 0.75         | 0.75                 |
| Top Performing Class (F1)   | nv - 0.91    | nv - 0.91            |

> **Clinical optimization:** Melanoma recall was prioritized to **reduce false negatives**, critical for patient safety.
---

## 🛠️ Tech Stack
- **Core:** PyTorch 2.0+ (MPS acceleration for Mac)  
- **Data Processing:** NumPy, Pandas (metadata handling), PIL (image operations)  
- **Analytics:** Scikit-learn (weighted metrics), Matplotlib/Seaborn  
---
