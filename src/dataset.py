# src/dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class KeypointsDataset(Dataset):
    def __init__(self, csv_path, keypoints_dir, selected_classes):
        """
        Args:
            csv_path: CSV 파일 경로 (예: "data/UCF101/train.csv")
            keypoints_dir: keypoints 폴더 경로 (예: "data/UCF101/keypoints/train")
            selected_classes: 사용할 클래스 리스트 (예: ["Basketball", "PushUps", ...])
        """
        self.keypoints_dir = keypoints_dir
        
        # Label mapping: 문자열 → 숫자
        self.label_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        
        # CSV 읽기
        df = pd.read_csv(csv_path)
        
        # 선택한 클래스만 필터링
        df = df[df['label'].isin(selected_classes)]
        
        # clip_name과 label 저장
        self.clip_names = df['clip_name'].tolist()
        self.labels = df['label'].map(self.label_to_idx).tolist()
        
        print(f"Dataset loaded: {len(self)} samples")
        print(f"Classes: {selected_classes}")
        
    def __len__(self):
        return len(self.clip_names)
    
    def __getitem__(self, idx):
        """
        Returns:
            keypoints: (30, 17, 3) FloatTensor
            label: int (0~4)
        """
        clip_name = self.clip_names[idx]
        label = self.labels[idx]
        
        # .npy 파일 로드
        keypoints_path = os.path.join(self.keypoints_dir, f"{clip_name}.npy")
        keypoints = np.load(keypoints_path)  # (30, 17, 3)
        
        # Tensor 변환
        keypoints = torch.FloatTensor(keypoints)
        label = torch.tensor(label, dtype=torch.long)
        
        return keypoints, label