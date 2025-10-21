# quantization/export_to_onnx.py
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('/workspace')
from src.model import PoseTransformerClassifier, PoseTCNClassifier
from src.dataset import KeypointsDataset

# ==================== Config ====================
DEVICE = "cpu"
NUM_CLASSES = 10

BASE_DIR = "/workspace/data/UCF101"
SELECTED_CLASSES = ["Skiing", "PushUps", "Punch", "Biking", "JumpRope",
                    "Diving", "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"]

CHECKPOINT_DIR = "/workspace/models/checkpoints"
CHECKPOINTS = {
    'transformer': os.path.join(CHECKPOINT_DIR, 'best_model_transformer.pth'),
    'tcn': os.path.join(CHECKPOINT_DIR, 'best_model_tcn.pth')
}

RESULTS_DIR = "/workspace/quantization/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ==============================================


class CalibrationDataReader:
    """Calibration data reader for static quantization"""
    def __init__(self, num_samples=100):
        print(f"  Loading calibration data ({num_samples} samples)...")
        
        dataset = KeypointsDataset(
            csv_path=os.path.join(BASE_DIR, "train.csv"),
            keypoints_dir=os.path.join(BASE_DIR, "keypoints/train"),
            selected_classes=SELECTED_CLASSES
        )
        
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        self.data = [dataset[i][0].numpy().astype(np.float32) for i in indices]
        self.current_index = 0
    
    def get_next(self):
        if self.current_index >= len(self.data):
            return None
        
        data = {'keypoints': np.expand_dims(self.data[self.current_index], axis=0)}
        self.current_index += 1
        return data
    
    def rewind(self):
        self.current_index = 0


def load_pytorch_model(model_type, checkpoint_path):
    """Load PyTorch model"""
    if model_type == 'transformer':
        model = PoseTransformerClassifier(
            input_size=51, d_model=128, nhead=4,
            num_layers=2, num_classes=NUM_CLASSES, dropout=0.1
        )
    elif model_type == 'tcn':
        model = PoseTCNClassifier(
            input_size=51, num_channels=[128, 256, 512],
            kernel_size=3, num_classes=NUM_CLASSES, dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 'N/A')
    if isinstance(val_acc, float):
        val_acc = f"{val_acc:.2f}%"
    
    print(f"[{model_type.upper()}] Loaded checkpoint (Val Acc: {val_acc})")
    return model


def export_to_onnx_fp32(model, model_name):
    """PyTorch -> ONNX FP32"""
    dummy_input = torch.randn(1, 30, 17, 3)
    onnx_fp32_path = os.path.join(RESULTS_DIR, f"{model_name}_fp32.onnx")
    
    torch.onnx.export(
        model, dummy_input, onnx_fp32_path,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['keypoints'], output_names=['logits'],
        dynamic_axes={'keypoints': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        verbose=False
    )
    
    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"  FP32 exported: {fp32_size:.2f} MB")
    return onnx_fp32_path


def quantize_onnx_to_int8_qdq(onnx_fp32_path, model_name):
    """
    ONNX FP32 -> INT8 Static Quantization (QDQ format)
    
    QDQ (Quantize-Dequantize):
    - QuantizeLinear + DequantizeLinear 쌍
    - CPU에서 실행 가능 (FP32로 fallback)
    - ConvInteger 사용 안 함
    """
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader as BaseReader
    
    print(f"\n[Step 2/2] Quantizing to INT8 (Static QDQ)...")
    
    onnx_int8_path = os.path.join(RESULTS_DIR, f"{model_name}_int8.onnx")
    
    # Calibration data reader
    calibration_reader = CalibrationDataReader(num_samples=100)
    
    # Static Quantization with QDQ format
    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,  # QDQ format (CPU 호환)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=False,  # Per-tensor (더 호환적)
        reduce_range=False
    )
    
    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    reduction = (1 - int8_size / fp32_size) * 100
    
    print(f"  INT8 quantized: {int8_size:.2f} MB (reduction: {reduction:.1f}%)")
    print(f"  Format: QDQ (CPU compatible)")
    return onnx_int8_path, reduction


def verify_onnx(onnx_path):
    """Verify ONNX model"""
    import onnx
    import onnxruntime as ort
    
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        dummy_input = np.random.randn(1, 30, 17, 3).astype(np.float32)
        outputs = session.run(None, {'keypoints': dummy_input})
        
        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def process_model(model_type, checkpoint_path):
    """Process single model"""
    try:
        # Load model
        model = load_pytorch_model(model_type, checkpoint_path)
        
        # Export to ONNX FP32
        onnx_fp32_path = export_to_onnx_fp32(model, model_type)
        if not verify_onnx(onnx_fp32_path):
            print(f"  FP32 verification failed")
            return None
        
        # Quantize to INT8 (QDQ)
        onnx_int8_path, reduction = quantize_onnx_to_int8_qdq(onnx_fp32_path, model_type)
        if not verify_onnx(onnx_int8_path):
            print(f"  INT8 verification failed")
            return None
        
        return {
            'fp32_path': onnx_fp32_path,
            'int8_path': onnx_int8_path,
            'fp32_size': os.path.getsize(onnx_fp32_path) / (1024 * 1024),
            'int8_size': os.path.getsize(onnx_int8_path) / (1024 * 1024),
            'reduction': reduction
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_summary(results):
    """Save summary to file"""
    summary_path = os.path.join(RESULTS_DIR, "onnx_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("ONNX Export & Quantization Summary (QDQ Format)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Model':<15} {'FP32 (MB)':<12} {'INT8 (MB)':<12} {'Reduction':<12}\n")
        f.write("-"*70 + "\n")
        
        for model_type, result in results.items():
            f.write(f"{model_type:<15} {result['fp32_size']:>10.2f} "
                   f"{result['int8_size']:>10.2f} {result['reduction']:>10.1f}%\n")
    
    print(f"\nSummary saved: {summary_path}")


def main():
    print("="*70)
    print("ONNX Export & Static Quantization (QDQ)")
    print("="*70)
    
    results = {}
    
    for model_type, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f"\n[{model_type.upper()}] Checkpoint not found, skipping")
            continue
        
        print(f"\n[{model_type.upper()}] Processing...")
        result = process_model(model_type, checkpoint_path)
        if result:
            results[model_type] = result
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"\n{'Model':<15} {'FP32 (MB)':<12} {'INT8 (MB)':<12} {'Reduction':<12}")
        print("-"*70)
        
        for model_type, result in results.items():
            print(f"{model_type:<15} {result['fp32_size']:>10.2f} "
                  f"{result['int8_size']:>10.2f} {result['reduction']:>10.1f}%")
        
        save_summary(results)
        print(f"\nResults saved to: {RESULTS_DIR}")
    else:
        print("\nNo models were exported")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()