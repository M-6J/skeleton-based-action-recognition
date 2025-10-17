# Skeleton-based Action Recognition with Quantization

YOLOv11-pose + Temporal Models (Transformer/TCN) for skeleton-based action recognition with ONNX/TensorRT quantization.

## Project Structure
```
├── data/
│ └── UCF101/
│ ├── train.csv
│ ├── val.csv
│ ├── test.csv
│ └── keypoints/
│   ├── train/.npy
│   ├── val/.npy
│   └── test/.npy
│
├── Dockerfile  # Container image definition
├── docker/
│ ├── build.sh  # Build Docker image
│ ├── run.sh    # Run container
│ └── root.sh   # Root access to container
│ 
├── models/
│ ├── yolo11x-pose.pt # Pretrained backbone
│ └── checkpoints/
│   ├── best_model_tcn.pth # FP32 trained model
│   └── best_model_transformer.pth # FP32 trained model
│
├── src/
│ ├── preprocessing.py # Extract keypoints from videos
│ ├── dataset.py # PyTorch Dataset
│ ├── model.py # Transformer/TCN classifier
│ ├── train.py # Training script
│ └── evaluate.py # Evaluation
│
├── quantization/
│ ├── export_to_onnx.py    # PyTorch → ONNX
│ ├── export_to_tensorrt.py # ONNX → TensorRT
│ ├── benchmark_all.py      # Performance comparison
│ └── results/  # Converted model results(ONNX, TensorRT)
│
└── env/requirements.txt
```

## Quick Start
### 1. Extract Keypoints
```python src/preprocessing.py```
### 2. Train
```python src/train.py --model tcn```
### 3. Evaluate
```python src/evaluate.py```
### 4. Export & Quantize
```
# ONNX export
python quantization/export_to_onnx.py
# TensorRT export
python quantization/export_to_tensorrt.py
```
### 5. Benchmark
```python quantization/benchmark_all.py```

## Results
### Dataset: UCF101(10 action classes)
```
TRANSFORMER
----------------------------------------------------------------------
Framework            Acc (%)    Size (MB)    Time (ms)    Speedup   
----------------------------------------------------------------------
PYTORCH FP32            88.55        7.30        0.54      1.00x
ONNX FP32               88.55        4.11        0.32      1.72x
ONNX INT8               87.95        3.02        0.35      1.55x
TRT FP32                88.55        2.48        0.29      1.86x
TRT FP16                88.55        1.32        0.25      2.14x

TCN
----------------------------------------------------------------------
Framework            Acc (%)    Size (MB)    Time (ms)    Speedup   
----------------------------------------------------------------------
PYTORCH FP32            90.36       21.27        1.12      1.00x
ONNX FP32               90.36        7.10        0.50      2.23x
ONNX INT8               90.36        1.84        0.44      2.55x
TRT FP32                90.36        7.91        0.37      3.02x
TRT FP16                90.36        5.12        0.34      3.26x
TRT INT8                90.36        2.50        0.33      3.40x
```