# Engineering Log: Optimizing PyTorch for Apple Silicon

This document serves as a technical post-mortem for the HAM10000 skin lesion classification project. It outlines the specific platform-level bottlenecks encountered when training deep learning models on macOS and the engineering decisions made to resolve them.

---

### 1. The I/O Tax: Solving the "10-Hour Epoch"
Initial training runs were projected to take over 10 hours per epoch. The bottleneck was identified not in the GPU kernels, but in the Python Global Interpreter Lock (GIL) and disk latency.

* **The Problem:** The `SkinLesionDataset.__getitem__` method was executing `os.path.exists()` for both `part_1` and `part_2` directories on every sample access.
* **The Root Cause:** Performing 20,000+ file system checks per epoch created a massive CPU overhead that kept the GPU idle while waiting for data.
* **The Fix:** Path resolution logic was moved to the `__init__` constructor. I cached the absolute paths for all 10,015 images into a dictionary at startup.
* **Performance Gain:** Training time was reduced from **hours to minutes**.

### 2. Multiprocessing: The `spawn` vs `fork` Constraint
On macOS, PyTorch defaults to the `spawn` start method for DataLoaders. This surfaced a critical failure when using `num_workers > 0`.



* **The Technical Hurdle:** Unlike Linux's `fork`, `spawn` starts a fresh Python interpreter. If the `Dataset` class is defined in the main script, the worker processes cannot "unpickle" the object because they lack the class definition in their fresh namespace.
* **Engineering Decision:** I decoupled the data pipeline by moving the `SkinLesionDataset` to a standalone module (`my_data_utils.py`) and enforced the `if __name__ == "__main__":` entry point. This ensured worker processes could successfully import the necessary dependencies.

### 3. Hardware Acceleration: MPS and the Unified Memory Fallacy
Transitioning from NVIDIA-centric tutorials to Apple Silicon required a shift in how hardware resources are managed.

* **Device Targeting:** PyTorch does not automatically route to the Mac GPU. I implemented an explicit check for the Metal Performance Shaders (MPS) backend:
    ```python
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ```
* **The pin_memory Mistake:** Habitual use of `pin_memory=True` (standard for CUDA) caused kernel instability. 
* **The Fix:** On Apple Silicon's **Unified Memory Architecture**, the CPU and GPU share the same physical RAM. Memory pinning is redundant and counterproductive. I explicitly disabled it to prevent Out-of-Memory (OOM) errors.


### 4. Data Persistence and State Isolation
A common pitfall in modular ML projects is the loss of training state across script boundaries. 

* **The Problem:** Training metrics (loss/accuracy) exist only in volatile memory. If the script ends, the "history" object dies.
* **The Solution:** I implemented a serialization layer. Training history is now exported to `.json` and model weights to `.pth` checkpoints. 
* **Why it Matters:** This allows the evaluation and visualization scripts to function as independent units, enabling post-hoc analysis without re-running 50+ epochs of training.

### 5. Defensive Engineering: Pre-Flight Validation
To mitigate the risk of mid-training crashes, I developed a "Pre-Flight" diagnostic suite. 

Before a single weight is updated, the system validates:
* **Path Integrity:** Confirms the path-resolution logic bridges both image directories correctly.
* **Metadata Alignment:** Ensures the 10,015 images match the CSV metadata exactly.
* **Tensor Geometry:** Validates that samples are $600 \times 450$ RGB, matching the input requirements for ResNet and EfficientNet.

---

### Summary of Performance Optimizations

| Feature | Standard Implementation | Optimized Implementation |
| :--- | :--- | :--- |
| **Path Discovery** | Real-time `os.path` checks | Pre-cached Path Mapping |
| **Backend** | CPU / CUDA (fails) | MPS (Metal Performance Shaders) |
| **Memory** | `pin_memory=True` | Unmanaged (Unified Memory) |
| **Data Loading** | Inline / `fork` | Modular / `spawn` |
| **Throughput** | ~1x (Baseline) | **10-20x Speedup** |

---