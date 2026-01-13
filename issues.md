# Challenges & Engineering Lessons Learned

This project involved training deep learning models (ResNet50 and EfficientNet) on the HAM10000 skin lesion dataset using PyTorch on macOS (Apple Silicon). While the model architectures themselves were standard, the project surfaced several **non-trivial engineering challenges** related to data pipelines, multiprocessing, file systems, hardware acceleration, and experiment reproducibility.

This document outlines the key challenges encountered, the root causes behind them, and the engineering decisions used to resolve each issue.

## 1. macOS Multiprocessing and the `spawn` Constraint
### Challenge  
On macOS, PyTorch’s DataLoader uses the **`spawn`** multiprocessing start method by default. Unlike Linux’s `fork`, `spawn` launches a completely fresh Python interpreter for each worker process.
Initially, my custom `SkinLesionDataset` class was defined directly inside the training script. When using `num_workers > 0`, DataLoader workers consistently crashed because they could not locate or unpickle the Dataset object.

### Root Cause  
With `spawn`, worker processes:
- Do not inherit the parent process memory
- Cannot access classes defined in the script’s main body
- Require an explicit entry point guarded by `if __name__ == "__main__":`
Without these conditions, the Dataset object could not be reconstructed in worker processes.

### Fix  
- Moved `SkinLesionDataset` into a standalone module (`my_data_utils.py`)
- Wrapped all training logic in an `if __name__ == "__main__":` block

### Why It Matters  
This is a **platform-specific multiprocessing failure mode** that frequently appears in real-world ML systems. Resolving it enabled stable multi-worker data loading and prevented silent crashes.

## 2. The `os.path.exists` Bottleneck (The 10-Hour Epoch)
### Challenge  
The HAM10000 dataset stores images across two directories:
- `HAM10000_images_part_1`
- `HAM10000_images_part_2`
Originally, the Dataset’s `__getitem__` method checked whether an image existed in part 1 or part 2 **for every sample access**.

### Root Cause  
This resulted in:
- 20,000+ file system checks per epoch
- Excessive disk I/O overhead
- Training epochs stretching from minutes to **hours**

### Fix  
- Moved all path-resolution logic to the Dataset’s `__init__`
- Cached resolved image paths once during initialization

### Why It Matters  
File system operations are expensive. This optimization reduced redundant I/O and was the **primary reason training time dropped from hours to minutes**, demonstrating the importance of optimizing the data pipeline, not just the model.

## 3. The `pin_memory` Fallacy on Apple Silicon
### Challenge  
Following standard PyTorch tutorials, `pin_memory=True` was initially enabled in the DataLoader.
### Root Cause  
Pinned memory improves data transfer on **NVIDIA GPUs**, but Apple Silicon uses a **Unified Memory Architecture**, where the CPU and GPU already share the same RAM.

On macOS, pinning memory:
- Provides no benefit
- Can cause out-of-memory errors or kernel instability

### Fix  
- Explicitly set `pin_memory=False` when using the MPS backend
### Why It Matters  
This highlights a common optimization pitfall. Hardware-aware tuning is critical, and blindly applying best practices from other platforms can degrade performance or stability.

## 4. `num_workers` and Dataset Visibility Failures
### Challenge  
Increasing `num_workers` repeatedly caused DataLoader crashes when the Dataset class was defined inside the training script.
### Root Cause  
macOS cannot `fork` process memory. Under `spawn`, each worker must re-import all dependencies. Inline class definitions are invisible to worker processes.

### Fix  
- Refactored Dataset and utility logic into importable modules
- Ensured multiprocessing-safe script structure
### Why It Matters  
This issue required understanding Python multiprocessing internals rather than PyTorch APIs alone, reinforcing the importance of systems-level knowledge in ML engineering.

## 5. Hardware Under-utilization (CPU vs. MPS)
### Challenge  
The initial device selection logic only checked for CUDA:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
