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

# Import the dataset from your existing helper file
from my_data_utils import SkinLesionDataset

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32  
NUM_WORKERS = 10
NUM_EPOCHS_PHASE1 = 15
NUM_EPOCHS_PHASE2 = 10

# Output paths
OUTPUT_DIR = "trained_model_file"
SAVE_FILENAME = os.path.join(OUTPUT_DIR, "effnet_baseline_running.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "effnet_b0_best.pth")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "skin_lesion_model.pth")

# Device selection (MPS for Mac GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f" Training EfficientNet-B0 on: {device}")

# --- MODEL FUNCTIONS ---
def create_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights='DEFAULT')
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model

def unfreeze_effnet_blocks(model, num_blocks=3):
    for param in model.features[-num_blocks:].parameters():
        param.requires_grad = True
    print(f"Unfrozen the last {num_blocks} blocks of the backbone.")

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
            
            pbar.set_postfix({'acc': f"{100.*correct/total:.2f}%"})
            
    return running_loss / total, 100. * correct / total

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Setup Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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

    # 3. Loaders
    train_dataset = SkinLesionDataset(train_df, data_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            persistent_workers=True)

    # 4. Initialize Model
    num_classes = len(train_df['label'].unique())
    model = create_efficientnet_b0(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # --- INITIALIZE HISTORY ---
    effnet_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # --- PHASE 1: CLASSIFIER ---
    print(f"\n{'='*20} PHASE 1: HEAD ONLY {'='*20}")
    best_acc = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS_PHASE1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        
        # Log Metrics
        effnet_history['train_loss'].append(t_loss)
        effnet_history['train_acc'].append(t_acc)
        effnet_history['val_loss'].append(v_loss)
        effnet_history['val_acc'].append(v_acc)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_PHASE1}] | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # --- PHASE 2: FINE-TUNING ---
    print(f"\n{'='*20} PHASE 2: FINE-TUNING {'='*20}")
    unfreeze_effnet_blocks(model, num_blocks=3)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    for epoch in range(NUM_EPOCHS_PHASE2):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        
        # Log Metrics
        effnet_history['train_loss'].append(t_loss)
        effnet_history['train_acc'].append(t_acc)
        effnet_history['val_loss'].append(v_loss)
        effnet_history['val_acc'].append(v_acc)
        
        current_total_epoch = NUM_EPOCHS_PHASE1 + epoch + 1
        print(f"Epoch [{current_total_epoch}/{NUM_EPOCHS_PHASE1+NUM_EPOCHS_PHASE2}] | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), 'effnet_b0_best.pth')

    # --- FINAL SAVE ---
    with open(SAVE_FILENAME, 'w') as f:
        json.dump(effnet_history, f, indent=4)

    print(f"\n Done! Total Time: {(time.time()-start_time)/60:.2f} mins")
    print(f" History saved to: {SAVE_FILENAME}")

        # --- FINAL SAVE WITH METADATA (add this at the very end) ---
    # Get actual class names from your dataframe
    class_to_idx = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

        # Get the actual class names (not just indices)
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # alphabetically sorted
    # OR get them from your dataframe:
    # class_names = sorted(train_df['label'].unique())

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': 'efficientnet_b0',
        'num_classes': num_classes,
        'class_names': class_names,  # Use actual names, not indices
        'input_size': IMG_SIZE,
        'best_accuracy': best_acc,
        'normalization_mean': [0.485, 0.456, 0.406],
        'normalization_std': [0.229, 0.224, 0.225]
    }

    os.makedirs('models', exist_ok=True)
    torch.save(checkpoint, 'trained_model_file/skin_lesion_model.pth')
