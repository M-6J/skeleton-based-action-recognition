# quantization/test_executorch.py
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/workspace')
from src.dataset import KeypointsDataset

from executorch.extension.pybindings.portable_lib import _load_for_executorch

RESULTS_DIR = "/workspace/quantization/results"
BASE_DIR = "/workspace/data/UCF101"
SELECTED_CLASSES = ["Skiing", "PushUps", "Punch", "Biking", "JumpRope",
                    "Diving", "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"]


class ExecuTorchInference:
    def __init__(self, pte_path):
        print(f"Loading: {pte_path}")
        self.module = _load_for_executorch(pte_path)
        self.is_gpu = False
    
    def predict(self, x):
        """x: numpy array (batch, 30, 17, 3)"""
        t = torch.from_numpy(x.astype(np.float32))
        out = self.module.forward([t])[0]
        return out.numpy()


def benchmark(predictor, dataloader, model_name):
    print(f"\n[{model_name.upper()} - ExecuTorch]")
    
    # Warm-up
    print("  Warming up...")
    for i, (keypoints, _) in enumerate(dataloader):
        if i >= 10:
            break
        keypoints_np = keypoints.numpy().astype(np.float32)
        _ = predictor.predict(keypoints_np)
    
    # Benchmark
    correct = 0
    total = 0
    inference_times = []
    
    for keypoints, labels in tqdm(dataloader, desc="  Testing"):
        keypoints_np = keypoints.numpy().astype(np.float32)
        
        # 3번 측정
        times = []
        for _ in range(3):
            start = time.perf_counter()
            outputs = predictor.predict(keypoints_np)
            end = time.perf_counter()
            times.append(end - start)
        
        inference_times.append(np.median(times))
        
        # Accuracy
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    
    accuracy = 100. * correct / total
    avg_time = np.mean(inference_times) * 1000  # ms
    
    return {
        'accuracy': accuracy,
        'inference_time': avg_time,
    }


def test_model(model_name, pte_path, test_loader):
    print(f"\n{'='*70}")
    print(f"{model_name.upper()}")
    print(f"{'='*70}")
    
    if not os.path.exists(pte_path):
        print(f"  File not found: {pte_path}")
        return None
    
    try:
        # Load
        predictor = ExecuTorchInference(pte_path)
        
        # Benchmark
        result = benchmark(predictor, test_loader, model_name)
        
        # Size
        size_mb = os.path.getsize(pte_path) / (1024 * 1024)
        result['size'] = size_mb
        
        print(f"\n  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Time: {result['inference_time']:.2f} ms")
        
        return result
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*70)
    print("ExecuTorch Inference Test")
    print("="*70)
    
    # Load dataset
    print("\nLoading test dataset...")
    test_dataset = KeypointsDataset(
        csv_path=os.path.join(BASE_DIR, "test.csv"),
        keypoints_dir=os.path.join(BASE_DIR, "keypoints/test"),
        selected_classes=SELECTED_CLASSES
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # Test models
    models = ['transformer', 'tcn']
    results = {}
    
    for model_name in models:
        pte_path = os.path.join(RESULTS_DIR, f"{model_name}_executorch.pte")
        result = test_model(model_name, pte_path, test_loader)
        if result:
            results[model_name] = result
    
    # Summary
    if results:
        print("\n\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Model':<15} {'Acc (%)':<10} {'Size (MB)':<12} {'Time (ms)':<12}")
        print("-"*70)
        
        for model_name, result in results.items():
            print(f"{model_name:<15} {result['accuracy']:>8.2f}  "
                  f"{result['size']:>10.2f}  {result['inference_time']:>10.2f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()