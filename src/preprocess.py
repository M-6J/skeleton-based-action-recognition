# 비디오 → Keypoints 추출 → .npy 저장
# 왜? = YOLO 추론은 느림 → 매 epoch마다 반복하면 비효율 → 한 번만 실행하고 저장

"""
    ① 비디오 열기
    ② 30프레임 균등 샘플링
       예: 180 프레임 → [0, 6, 12, ..., 174]
    ③ 각 프레임마다:
       - YOLO로 사람 감지
       - Keypoints 추출 (17개 관절)
       - 정규화된 좌표 (0~1)
    ④ 결과: (30, 17, 3) numpy 배열
       - 30: 프레임 수
       - 17: keypoints (코, 눈, 어깨 등)
       - 3: (x, y, confidence)
"""

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ==================== 설정 (여기서 직접 수정) ====================
BASE_DIR = "data/UCF101"
YOLO_MODEL_PATH = "models/yolo11x-pose.pt"
SELECTED_CLASSES = ["Skiing", "PushUps", "Punch", "Biking", "JumpRope", "Diving",
                     "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"]
NUM_FRAMES = 30
# ==================================================================

def extract_keypoints_from_video(video_path, model, num_frames=30): #30프레임 균등 샘플링
    """
    비디오에서 균등하게 샘플링된 프레임의 keypoints 추출 (Batch inference)
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # 균등 샘플링
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    # 프레임 읽기 (한번에)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    cap.release()
    
    # Batch inference
    results = model(frames, verbose=False)
    
    # Keypoints 추출
    keypoints_list = []
    for result in results:
        if len(result.keypoints.data) > 0:
            # 여러 사람이면 가장 큰 사람
            if len(result.boxes.xyxy) > 1:
                boxes = result.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_idx = areas.argmax()
                
                # 정규화된 좌표 (0~1) 사용
                kpts_xy = result.keypoints.xyn[max_idx].cpu().numpy()  # (17, 2)
                kpts_conf = result.keypoints.conf[max_idx].cpu().numpy()  # (17,)
            else:
                # 정규화된 좌표 (0~1) 사용
                kpts_xy = result.keypoints.xyn[0].cpu().numpy()  # (17, 2)
                kpts_conf = result.keypoints.conf[0].cpu().numpy()  # (17,)
            
            # (17, 2)와 (17, 1)을 합쳐서 (17, 3) 만들기
            kpts = np.concatenate([kpts_xy, kpts_conf[:, None]], axis=1)
            keypoints_list.append(kpts)
        else:
            keypoints_list.append(np.zeros((17, 3)))
    
    return np.array(keypoints_list)


def preprocess_split(csv_path, split_name):
    """
    한 split(train/val/test)의 비디오를 전처리
    """
    print(f"\n{'='*50}")
    print(f"Processing {split_name} set")
    print(f"{'='*50}")
    
    # CSV 읽기
    df = pd.read_csv(csv_path)
    print(f"Total videos in CSV: {len(df)}")
    
    # 선택한 클래스만 필터링
    df = df[df['label'].isin(SELECTED_CLASSES)]
    print(f"Filtered videos: {len(df)} (classes: {SELECTED_CLASSES})")
    
    if len(df) == 0:
        print(f"Warning: No videos found for selected classes in {split_name}")
        return
    
    # YOLO 모델 로드
    print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    model.to('cuda')
    # 저장 디렉토리 생성
    save_dir = os.path.join(BASE_DIR, "keypoints", split_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # 각 비디오 처리
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        clip_name = row['clip_name']
        clip_path = row['clip_path']
        
        # 경로 처리: /train/Biking/v_Biking.avi → UCF101/train/Biking/v_Biking.avi
        if clip_path.startswith('/'):
            clip_path = clip_path[1:]
        video_path = os.path.join(BASE_DIR, clip_path)
        
        # 비디오 존재 확인
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            fail_count += 1
            continue
        
        # Keypoints 추출
        keypoints = extract_keypoints_from_video(video_path, model, NUM_FRAMES)
        
        if keypoints is None:
            print(f"Failed to extract keypoints: {clip_name}")
            fail_count += 1
            continue
        
        # 저장
        save_path = os.path.join(save_dir, f"{clip_name}.npy")
        np.save(save_path, keypoints)
        success_count += 1
    
    print(f"\n{split_name} completed:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")

if __name__ == "__main__":
    print("="*50)
    print("Keypoints Extraction Preprocessing")
    print("="*50)
    print(f"Base directory: {BASE_DIR}")
    print(f"Selected classes: {SELECTED_CLASSES}")
    print(f"Number of frames per video: {NUM_FRAMES}")
    
    # Train, Val, Test 처리
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(BASE_DIR, f"{split}.csv")
        if os.path.exists(csv_path):
            preprocess_split(csv_path, split)
        else:
            print(f"\nWarning: {csv_path} not found, skipping {split} set")
    
    print("\n" + "="*50)
    print("Preprocessing completed!")
    print("="*50)