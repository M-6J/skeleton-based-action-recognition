# src/evaluate.py
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from dataset import KeypointsDataset
from model import PoseTransformerClassifier, PoseTCNClassifier

# ==================== 설정 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (Docker 환경)
WORKSPACE = "/workspace"
BASE_DIR = os.path.join(WORKSPACE, "data/UCF101")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "models/checkpoints")

# 10개 클래스
SELECTED_CLASSES = [
    "Skiing", "PushUps", "Punch", "Biking", "JumpRope",
    "Diving", "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"
]

# 체크포인트 경로
CHECKPOINTS = {
    'transformer': os.path.join(CHECKPOINT_DIR, 'best_model_former.pth'),
    'tcn': os.path.join(CHECKPOINT_DIR, 'best_model_tcn.pth')
}
# ==============================================


def load_model(model_type, checkpoint_path):
    """모델 로드"""
    print(f"\nLoading {model_type.upper()} model...")
    
    # 모델 생성
    if model_type == 'transformer':
        model = PoseTransformerClassifier(
            input_size=51,
            d_model=128,
            nhead=4,
            num_classes=len(SELECTED_CLASSES),
            dropout=0.1
        )
    elif model_type == 'tcn':
        model = PoseTCNClassifier(
            input_size=51,
            num_channels=[128, 256, 512],
            kernel_size=3,
            num_classes=len(SELECTED_CLASSES),
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"   Loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Acc: {checkpoint['val_acc']:.2f}%")
    
    return model


def evaluate_model(model, dataloader, device, class_names, model_name):
    """상세 평가"""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*70}")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for keypoints, labels in dataloader:
            keypoints = keypoints.to(device)
            outputs = model(keypoints)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 1. Overall Accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\n📊 Overall Accuracy: {accuracy:.2f}%")
    
    # 2. Classification Report
    print(f"\n{'='*70}")
    print("Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    ))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n Confusion matrix saved: {save_path}")
    
    # 4. Per-class Accuracy
    print(f"\n{'='*70}")
    print("Per-class Accuracy")
    print(f"{'='*70}")
    print(f"{'Class':<20} {'Accuracy':>10} {'Samples':>10}")
    print("-"*70)
    
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean() * 100
            print(f"{class_name:<20} {class_acc:>9.2f}% {class_mask.sum():>10}")
        else:
            print(f"{class_name:<20} {'N/A':>10} {0:>10}")
    
    return accuracy, cm


def evaluate_all_models():
    """모든 모델 평가"""
    print("="*70)
    print("Model Evaluation")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Classes: {SELECTED_CLASSES}")
    print(f"Number of classes: {len(SELECTED_CLASSES)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test dataset
    print("\nLoading test dataset...")
    test_dataset = KeypointsDataset(
        csv_path=os.path.join(BASE_DIR, "test.csv"),
        keypoints_dir=os.path.join(BASE_DIR, "keypoints/test"),
        selected_classes=SELECTED_CLASSES
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    results = {}
    
    for model_type, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f"\n Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping {model_type}")
            continue
        
        try:
            # 모델 로드
            model = load_model(model_type, checkpoint_path)
            
            # 평가
            accuracy, cm = evaluate_model(
                model, test_loader, DEVICE, 
                SELECTED_CLASSES, model_type
            )
            
            results[model_type] = {
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"\n Error evaluating {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 결과 요약
    if results:
        print(f"\n\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Test Accuracy':>15}")
        print("-"*70)
        
        for model_type, result in results.items():
            print(f"{model_type:<20} {result['accuracy']:>14.2f}%")
        
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n Best Model: {best_model[0].upper()} ({best_model[1]['accuracy']:.2f}%)")
    
    print(f"\n{'='*70}")
    print(" Evaluation completed!")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    evaluate_all_models()