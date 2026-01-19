import os
import time
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# Local imports
try:
    from my_data_utils import SkinLesionDataset
except ImportError:
    raise ImportError("Could not import 'SkinLesionDataset' from 'my_data_utils.py'. Ensure the file exists.")

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
@dataclass
class TrainingConfig:
    """Central configuration for training hyperparameters."""
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs_phase1: int = 25  # Head training
    epochs_phase2: int = 10  # Fine-tuning
    
    # Paths (Relative is better for portability)
    data_dir: Path = Path("data/ham10000")
    csv_train: Path = Path("data/data/train_split.csv")
    csv_val: Path = Path("data/data/val_split.csv")
    output_dir: Path = Path("artifacts/checkpoints")
    
    device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- UTILITIES ---
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting well-classified examples.
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Initializes ConvNeXt-Base and modifies the classifier head."""
    model = models.convnext_base(weights='DEFAULT')
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace head
    # ConvNeXt classifier structure: [2] is the final Linear layer
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model

def unfreeze_layers(model: nn.Module, num_layers: int = 3) -> None:
    """Unfreezes the last N layers of the backbone for fine-tuning."""
    for param in model.features[-num_layers:].parameters():
        param.requires_grad = True
    logger.info(f"Unfrozen last {num_layers} layers for fine-tuning.")

def save_checkpoint(state: Dict, path: Path, is_best: bool = False) -> None:
    """Saves model state and creates a copy if it's the best model."""
    torch.save(state, path)
    if is_best:
        logger.info(f"New best model saved to {path}")

# --- TRAINING ENGINE ---
def run_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None, 
    is_train: bool = True
) -> Tuple[float, float]:
    """Runs a single epoch of training or validation."""
    model.train() if is_train else model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for progress tracking
    desc = 'Training' if is_train else 'Validation'
    pbar = tqdm(loader, desc=desc, leave=False)
    
    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if is_train:
                if optimizer is None:
                    raise ValueError("Optimizer cannot be None during training.")
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'acc': f"{100.*correct/total:.2f}%"})
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_pipeline(cfg: TrainingConfig, class_names: List[str]):
    """Orchestrates the two-phase training process."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Data Setup
    logger.info("Initializing DataLoaders...")
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Assumes CSVs exist at the config paths
    try:
        train_df = pd.read_csv(cfg.csv_train)
        val_df = pd.read_csv(cfg.csv_val)
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        return

    train_loader = DataLoader(
        SkinLesionDataset(train_df, cfg.data_dir, transform=train_transform),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, persistent_workers=True
    )
    val_loader = DataLoader(
        SkinLesionDataset(val_df, cfg.data_dir, transform=val_transform),
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, persistent_workers=True
    )

    # 2. Model Setup
    logger.info(f"Initializing ConvNeXt-Base on {cfg.device}")
    model = create_model(len(class_names)).to(cfg.device)
    criterion = FocalLoss(alpha=1, gamma=2)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_model_path = cfg.output_dir / "best_model.pth"

    # 3. Training Loop
    # Phase 1: Head Only
    logger.info(f"--- Phase 1: Training Head ({cfg.epochs_phase1} Epochs) ---")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    
    total_epochs = cfg.epochs_phase1 + cfg.epochs_phase2
    
    for epoch in range(total_epochs):
        # Phase switch logic
        if epoch == cfg.epochs_phase1:
            logger.info(f"--- Phase 2: Fine-Tuning ({cfg.epochs_phase2} Epochs) ---")
            unfreeze_layers(model)
            # Lower LR for fine-tuning
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)

        # Run Epoch
        t_loss, t_acc = run_epoch(model, train_loader, criterion, cfg.device, optimizer, is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, cfg.device, None, is_train=False)
        
        # Logging
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        logger.info(f"Epoch [{epoch+1}/{total_epochs}] | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")
        
        # Save Best
        if v_acc > best_acc:
            best_acc = v_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': vars(cfg),
                'best_accuracy': v_acc,
                'epoch': epoch + 1
            }
            save_checkpoint(checkpoint, best_model_path, is_best=True)

    # 4. Finalize
    logger.info(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")
    with open(cfg.output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=4)
        
    return best_model_path

def deploy_model(source_path: Path, deploy_dirs: List[Path]):
    """Copies the best model artifact to deployment directories."""
    if not source_path.exists():
        logger.error("Source model not found, skipping deployment copy.")
        return

    logger.info("Deploying model artifacts...")
    for d in deploy_dirs:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, d / "skin_lesion_model.pth")
        logger.info(f" -> Copied to {d}")

if __name__ == '__main__':
    # Initialize Config
    config = TrainingConfig()
    
    # Load class names (Needed for model creation)
    # Ideally, this should come from a metadata file, but we read it from CSV for now
    try:
        df = pd.read_csv(config.csv_train)
        classes = sorted(df['label'].unique())
    except Exception as e:
        logger.error(f"Failed to read classes from CSV: {e}")
        exit(1)

    # Run Training
    best_path = train_pipeline(config, classes)
    
    # Run Deployment
    deploy_targets = [
        config.output_dir,
        Path("gradio_app/model")
    ]
    deploy_model(best_path, deploy_targets)