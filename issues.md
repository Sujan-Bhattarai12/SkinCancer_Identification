# Engineering Lessons Learned: Deep Learning on Apple Silicon

This document details the technical challenges and engineering decisions made while training ResNet50 and EfficientNet on the HAM10000 dataset using PyTorch and macOS (MPS).

---

## 1. Performance and Hardware Optimization

### Eliminating the os.path.exists Bottleneck
* **The Problem:** Training was taking 10+ hours per epoch. The dataset was split across two directories, and the `__getitem__` method performed a disk check for every single image access.
* **The Root Cause:** This resulted in over 20,000 redundant file system calls per epoch.
* **The Fix:** Path resolution was moved to the `__init__` method. All image paths are now resolved once and cached in memory during dataset initialization.
* **Impact:** Reduced epoch time from hours to minutes.

### Enabling Metal Performance Shaders (MPS)
* **The Problem:** Initial training defaulted to CPU, despite the availability of GPU cores on the M-series chip.
* **The Fix:** Implemented explicit device selection for the Apple Silicon backend:
    ```python
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ```
* **Impact:** Achieved a 10–20x speedup in training throughput compared to CPU execution.

### The pin_memory Fallacy
* **The Insight:** Standard PyTorch tutorials suggest `pin_memory=True` for speed. However, on Apple Silicon’s Unified Memory Architecture, the CPU and GPU share the same RAM pool.

* **The Fix:** Explicitly set `pin_memory=False`. Memory pinning provides no benefit in this architecture and can trigger kernel instability or out-of-memory errors on macOS.

---

## 2. Systems and Multiprocessing

### Resolving macOS spawn Constraints
* **The Problem:** DataLoader workers crashed consistently when `num_workers > 0`.
* **The Root Cause:** Unlike Linux (which uses `fork`), macOS uses `spawn`. Worker processes start a fresh interpreter and cannot access classes defined inline in the main script.

* **The Fix:** 1. Refactored the `SkinLesionDataset` class into a standalone module (`my_data_utils.py`).
    2. Wrapped the training execution logic in an `if __name__ == "__main__":` block.
* **Impact:** Enabled stable, high-speed multi-core data loading.

---

## 3. Experiment Management and Integrity

### Decoupling Training from Evaluation
* **The Problem:** Loss curves and accuracy metrics were lost once a script execution finished, hindering long-term analysis.
* **The Fix:** Implemented a persistence layer where model weights are saved as `.pth` checkpoints and training history is exported to `.json` files.
* **Impact:** Enforced reproducibility and allowed for visualization and analysis without re-running the heavy compute phase.

### Defensive Pre-Flight Validation
* **The Strategy:** Before initiating training, a diagnostic script validates the entire environment.
* **Checks Performed:**
    * Path resolution across split directories.
    * Dimensionality consistency (600 × 450).
    * Color mode verification (RGB).
* **Impact:** Prevented mid-training crashes and guaranteed data integrity before consuming compute resources.

---

## 4. Key Takeaways

1.  **I/O Overheads:** Data pipeline inefficiencies often dominate training time more than the model architecture itself.
2.  **Hardware-Specific Logic:** Optimizations designed for NVIDIA/Linux are not always applicable to Apple Silicon.
3.  **Modular Architecture:** Moving from script-based workflows to modular, importable code is required to utilize multiprocessing effectively on macOS.