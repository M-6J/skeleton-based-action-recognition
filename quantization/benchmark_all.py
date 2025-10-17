#!/usr/bin/env python3
# quantization/benchmark_all.py
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/workspace')
from src.dataset import KeypointsDataset
from src.model import PoseTransformerClassifier, PoseTCNClassifier

# ONNX Runtime
import onnxruntime as ort

# TensorRT
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ==================== Config ====================
RESULTS_DIR = "/workspace/quantization/results"
TRAIN_RESULTS_DIR = "/workspace/models/checkpoints"
BASE_DIR = "/workspace/data/UCF101"
SELECTED_CLASSES = ["Skiing", "PushUps", "Punch", "Biking", "JumpRope",
                    "Diving", "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"]
BATCH_SIZE = 1
# ==============================================


class PyTorchInference:
    """PyTorch model inference"""
    def __init__(self, model_path, model_type, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 구조 생성
        if model_type == 'transformer':
            self.model = PoseTransformerClassifier(
                input_size=51, d_model=128, nhead=4,
                num_layers=2, num_classes=10, dropout=0.1
            )
        elif model_type == 'tcn':
            self.model = PoseTCNClassifier(
                input_size=51, num_channels=[128, 256, 512],
                kernel_size=3, num_classes=10, dropout=0.2
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # state_dict 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        self.is_gpu = self.device.type == 'cuda'

    def predict(self, x):
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            out = self.model(t)
            return out.detach().cpu().numpy()


class ONNXInference:
    """ONNX Runtime Inference"""
    def __init__(self, onnx_path, use_gpu=True):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.is_gpu = 'CUDA' in self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, x):
        outputs = self.session.run(None, {self.input_name: x})
        return outputs[0]


class TensorRTInference:
    """TensorRT Inference"""
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.output_shape = (1, 10)
        self.stream = cuda.Stream()
        self.d_input = None
        self.d_output = None
        self.d_input_size = 0
        self.d_output_size = 0
    
    def predict(self, x):
        batch_size = x.shape[0]
        self.context.set_input_shape("keypoints", x.shape)
        
        input_size = int(x.nbytes)
        output_size = int(batch_size * 10 * np.dtype(np.float32).itemsize)
        
        if self.d_input is None or self.d_input_size != input_size:
            if self.d_input is not None:
                self.d_input.free()
            self.d_input = cuda.mem_alloc(input_size)
            self.d_input_size = input_size
        
        if self.d_output is None or batch_size != self.output_shape[0]:
            if self.d_output is not None:
                self.d_output.free()
            self.output_shape = (batch_size, 10)
            self.d_output = cuda.mem_alloc(output_size)
            self.d_output_size = output_size
        
        h_output = np.empty(self.output_shape, dtype=np.float32)
        
        cuda.memcpy_htod_async(self.d_input, x.astype(np.float32), self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return h_output


def benchmark_model(predictor, dataloader, model_name, framework):
    device_info = ""
    if hasattr(predictor, 'is_gpu'):
        device_info = " [GPU]" if predictor.is_gpu else " [CPU]"
    
    print(f"\n[{model_name.upper()} - {framework}{device_info}]")
    
    # Warm-up
    warmup_count = 0
    for keypoints, _ in dataloader:
        keypoints_np = keypoints.numpy().astype(np.float32)
        _ = predictor.predict(keypoints_np)
        if hasattr(predictor, 'stream'):
            cuda.Context.synchronize()
        warmup_count += 1
        if warmup_count >= 10:
            break
    
    # Benchmark
    correct = 0
    total = 0
    inference_times = []
    
    for keypoints, labels in tqdm(dataloader, desc=f"  {framework}"):
        keypoints_np = keypoints.numpy().astype(np.float32)
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            outputs = predictor.predict(keypoints_np)
            if hasattr(predictor, 'stream'):
                cuda.Context.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        inference_times.append(np.median(times))
        
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    
    accuracy = 100. * correct / total
    avg_time = np.mean(inference_times) * 1000
    
    return {
        'accuracy': accuracy,
        'inference_time': avg_time,
    }


def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return None


def run_benchmark():
    print("="*70)
    print("Quantization Benchmark")
    print("="*70)
    
    test_dataset = KeypointsDataset(
        csv_path=os.path.join(BASE_DIR, "test.csv"),
        keypoints_dir=os.path.join(BASE_DIR, "keypoints/test"),
        selected_classes=SELECTED_CLASSES
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    models = ['transformer', 'tcn']
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"{model_name.upper()}")
        print(f"{'='*70}")
        
        results = {}
        
        # PyTorch FP32
        pytorch_fp32_path = os.path.join(TRAIN_RESULTS_DIR, f"best_model_{model_name}_50.pth")
        if not os.path.exists(pytorch_fp32_path):
            pytorch_fp32_path = os.path.join(TRAIN_RESULTS_DIR, f"best_model_{model_name}.pth")
        
        if os.path.exists(pytorch_fp32_path):
            try:
                predictor = PyTorchInference(pytorch_fp32_path, model_type=model_name)
                result = benchmark_model(predictor, test_loader, model_name, "PyTorch FP32")
                result['size'] = get_file_size(pytorch_fp32_path)
                results['pytorch_fp32'] = result
            except Exception as e:
                print(f"  PyTorch FP32 failed: {e}")
        
        # ONNX FP32
        onnx_fp32_path = os.path.join(RESULTS_DIR, f"{model_name}_fp32.onnx")
        if os.path.exists(onnx_fp32_path):
            predictor = ONNXInference(onnx_fp32_path, use_gpu=True)
            result = benchmark_model(predictor, test_loader, model_name, "ONNX FP32")
            result['size'] = get_file_size(onnx_fp32_path)
            results['onnx_fp32'] = result
        
        # ONNX INT8
        onnx_int8_path = os.path.join(RESULTS_DIR, f"{model_name}_int8.onnx")
        if os.path.exists(onnx_int8_path):
            try:
                predictor = ONNXInference(onnx_int8_path, use_gpu=True)
                result = benchmark_model(predictor, test_loader, model_name, "ONNX INT8")
                result['size'] = get_file_size(onnx_int8_path)
                results['onnx_int8'] = result
            except Exception as e:
                print(f"  ONNX INT8 failed: {e}")
                results['onnx_int8'] = {'accuracy': None, 'inference_time': None, 'size': get_file_size(onnx_int8_path)}
        
        # TensorRT FP32
        trt_fp32_path = os.path.join(RESULTS_DIR, f"{model_name}_fp32.engine")
        if os.path.exists(trt_fp32_path):
            try:
                predictor = TensorRTInference(trt_fp32_path)
                result = benchmark_model(predictor, test_loader, model_name, "TensorRT FP32")
                result['size'] = get_file_size(trt_fp32_path)
                results['trt_fp32'] = result
            except Exception as e:
                print(f"  TensorRT FP32 failed: {e}")
        
        # TensorRT FP16
        trt_fp16_path = os.path.join(RESULTS_DIR, f"{model_name}_fp16.engine")
        if os.path.exists(trt_fp16_path):
            try:
                predictor = TensorRTInference(trt_fp16_path)
                result = benchmark_model(predictor, test_loader, model_name, "TensorRT FP16")
                result['size'] = get_file_size(trt_fp16_path)
                results['trt_fp16'] = result
            except Exception as e:
                print(f"  TensorRT FP16 failed: {e}")
        
        # TensorRT INT8 (TCN only)
        if model_name == 'tcn':
            trt_int8_path = os.path.join(RESULTS_DIR, f"{model_name}_int8.engine")
            if os.path.exists(trt_int8_path):
                try:
                    predictor = TensorRTInference(trt_int8_path)
                    result = benchmark_model(predictor, test_loader, model_name, "TensorRT INT8")
                    result['size'] = get_file_size(trt_int8_path)
                    results['trt_int8'] = result
                except Exception as e:
                    print(f"  TensorRT INT8 failed: {e}")
        
        all_results[model_name] = results
    
    return all_results


def print_summary(all_results):
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}")
        print("-"*70)
        print(f"{'Framework':<20} {'Acc (%)':<10} {'Size (MB)':<12} {'Time (ms)':<12} {'Speedup':<10}")
        print("-"*70)
        
        # Baseline: PyTorch FP32 > ONNX FP32 > first available
        baseline_key = None
        if 'pytorch_fp32' in results and results['pytorch_fp32'].get('inference_time'):
            baseline_key = 'pytorch_fp32'
        elif 'onnx_fp32' in results and results['onnx_fp32'].get('inference_time'):
            baseline_key = 'onnx_fp32'
        else:
            for k, v in results.items():
                if v.get('inference_time'):
                    baseline_key = k
                    break
        
        baseline_time = results[baseline_key]['inference_time'] if baseline_key else None
        
        for framework, result in results.items():
            acc = result.get('accuracy')
            size = result.get('size')
            time_ms = result.get('inference_time')
            
            framework_name = framework.replace('_', ' ').upper()
            
            if acc is None or time_ms is None:
                print(f"{framework_name:<20} {'N/A':<10} {size if size else 'N/A':>10}  {'N/A':<12} {'N/A':<10}")
                continue
            
            if baseline_time and baseline_time > 0:
                speedup = baseline_time / time_ms
            else:
                speedup = 1.0
            
            print(f"{framework_name:<20} {acc:>8.2f}  {size:>10.2f}  {time_ms:>10.2f}  {speedup:>8.2f}x")


def save_results(all_results):
    rows = []
    for model_name, results in all_results.items():
        for framework, result in results.items():
            rows.append({
                'model': model_name,
                'framework': framework,
                'accuracy': result.get('accuracy'),
                'size_mb': result.get('size'),
                'inference_ms': result.get('inference_time'),
            })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")


def main():
    results = run_benchmark()
    print_summary(results)
    save_results(results)
    print("\n" + "="*70)


if __name__ == "__main__":
    main()