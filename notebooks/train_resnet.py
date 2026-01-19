import os
import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm

# IMPORTANT: Import the dataset from our helper file
from my_data_utils import SkinLesionDataset

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4 
NUM_EPOCHS_PHASE1 = 1
NUM_EPOCHS_PHASE2 = 1

#output paths
OUTPUT_DIR = "trained_model_file"
SAVE_FILENAME = os.path.join(OUTPUT_DIR, "resnet_baseline_running.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "resnet50_best.pth")

# Set Device: prioritizing Mac GPU (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f" Using device: {device}")

def create_resnet50(num_classes):
    model = models.resnet50(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc='Training' if is_train else 'Validation', leave=False)
    
    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if is_train:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
    return running_loss / total, 100. * correct / total

if __name__ == '__main__':
    # 1. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Paths
    data_dir = '/Users/sujanbhattarai/deep_learning/data/ham10000/'
    train_df = pd.read_csv('/Users/sujanbhattarai/deep_learning/data/data/train_split.csv')
    val_df = pd.read_csv('/Users/sujanbhattarai/deep_learning/data/data/val_split.csv')

    # 3. Datasets & Loaders
    train_dataset = SkinLesionDataset(train_df, data_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, data_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )

    # 4. Model Setup
    num_classes = len(train_df['label'].unique())
    model = create_resnet50(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # --- NEW: INITIALIZE HISTORY DICTIONARY ---
    resnet_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # 5. Training Phase 1: Classifier Only
    print(f"\n{'='*30}\nPHASE 1: CLASSIFIER ONLY\n{'='*30}")
    best_acc = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS_PHASE1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        
        # --- NEW: LOG DATA ---
        resnet_history['train_loss'].append(t_loss)
        resnet_history['train_acc'].append(t_acc)
        resnet_history['val_loss'].append(v_loss)
        resnet_history['val_acc'].append(v_acc)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE1}] | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # 6. Training Phase 2: Fine-Tuning
    print(f"\n{'='*30}\nPHASE 2: FINE-TUNING\n{'='*30}")
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    for epoch in range(NUM_EPOCHS_PHASE2):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        
        # --- NEW: LOG DATA ---
        resnet_history['train_loss'].append(t_loss)
        resnet_history['train_acc'].append(t_acc)
        resnet_history['val_loss'].append(v_loss)
        resnet_history['val_acc'].append(v_acc)
        
        print(f"Epoch [{NUM_EPOCHS_PHASE1+epoch+1}/{NUM_EPOCHS_PHASE1+NUM_EPOCHS_PHASE2}] | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")

    # 7. Final Save
    total_time = (time.time() - start_time) / 60
    with open(SAVE_FILENAME, 'w') as f:
        json.dump(resnet_history, f, indent=4)
        
    print(f"\nTraining Complete! Total time: {total_time:.2f} minutes")
    print(f"History saved to {SAVE_FILENAME}")