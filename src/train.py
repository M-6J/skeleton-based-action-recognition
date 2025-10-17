# src/train.py
import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KeypointsDataset
from model import PoseTCNClassifier, PoseMSTCNClassifier, PoseTransformerClassifier
import argparse
# ==================== 설정 ====================
BASE_DIR = "data/UCF101"
SELECTED_CLASSES = ["Skiing", "PushUps", "Punch", "Biking", "JumpRope", "Diving",
                     "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"]

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

d_model = 128
NUM_LAYERS = 2

CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ==============================================

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model',
                    type=str,
                    default='mstcn',
                    help='select model: transformer, tcn, mstcn')
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help='random seed for reproducibility')
args = parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for keypoints, labels in pbar:
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for keypoints, labels in tqdm(dataloader, desc="Validation"):
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    print("="*50)
    print("Pose-based Action Recognition Training")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Classes: {SELECTED_CLASSES}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("="*50)
    
    # Dataset
    print("\n[1/5] Loading datasets...")
    train_dataset = KeypointsDataset(
        csv_path=os.path.join(BASE_DIR, "train.csv"),
        keypoints_dir=os.path.join(BASE_DIR, "keypoints/train"),
        selected_classes=SELECTED_CLASSES
    )
    
    val_dataset = KeypointsDataset(
        csv_path=os.path.join(BASE_DIR, "val.csv"),
        keypoints_dir=os.path.join(BASE_DIR, "keypoints/val"),
        selected_classes=SELECTED_CLASSES
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    print("\n[2/5] Creating model...")
    if args.model == 'transformer': 
        model = PoseTransformerClassifier(
            input_size=51,
            d_model=d_model,
            num_classes=len(SELECTED_CLASSES)
        ).to(DEVICE)
    elif args.model == 'tcn': 
        model = PoseTCNClassifier(
            input_size=51,
            num_classes=len(SELECTED_CLASSES)
        ).to(DEVICE)
    elif args.model == 'mstcn': 
        model = PoseMSTCNClassifier(
            input_size=51,
            num_classes=len(SELECTED_CLASSES)
        ).to(DEVICE)
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss, Optimizer, Scheduler
    print("\n[3/5] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    
    # Training
    print("\n[4/5] Starting training...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Scheduler step
        scheduler.step()
        
        # Results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{args.model}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f" Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("\n[5/5] Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved at: {CHECKPOINT_DIR}/best_model_{args.model}.pth")


if __name__ == "__main__":
    set_seed(args.seed)
    main()