#!/usr/bin/env python3
# quantization/export_to_executorch.py
import os
import sys
import torch
from torch.export import export

sys.path.append('/workspace')
from src.model import PoseTransformerClassifier, PoseTCNClassifier

from executorch.exir import to_edge, ExecutorchBackendConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

CHECKPOINT_DIR = "/workspace/models/checkpoints"
RESULTS_DIR = "/workspace/quantization/results"


def load_model(model_type):
    """Load PyTorch model"""
    if model_type == 'transformer':
        model = PoseTransformerClassifier(
            input_size=51, d_model=128, nhead=4,
            num_layers=2, num_classes=10, dropout=0.1
        )
    elif model_type == 'tcn':
        model = PoseTCNClassifier(
            input_size=51, num_channels=[128, 256, 512],
            kernel_size=3, num_classes=10, dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{model_type}_50.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{model_type}.pth")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def export_fp32(model, example_input, model_type):
    """Export FP32 model"""
    aten_dialect = export(model, example_input, strict=True)
    edge_program = to_edge(aten_dialect)
    executorch_program = edge_program.to_executorch()
    
    save_path = os.path.join(RESULTS_DIR, f"{model_type}_executorch_fp32.pte")
    with open(save_path, "wb") as f:
        f.write(executorch_program.buffer)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  FP32: {size_mb:.2f} MB")
    return save_path


def export_fp16(model, example_input, model_type):
    """Export FP16 model"""
    # Convert model to FP16
    model_fp16 = model.half()
    example_input_fp16 = tuple(x.half() for x in example_input)
    
    # Export with FP16 input
    aten_dialect = export(model_fp16, example_input_fp16, strict=False)  # strict=False for FP16
    edge_program = to_edge(aten_dialect)
    executorch_program = edge_program.to_executorch()
    
    save_path = os.path.join(RESULTS_DIR, f"{model_type}_executorch_fp16.pte")
    with open(save_path, "wb") as f:
        f.write(executorch_program.buffer)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  FP16: {size_mb:.2f} MB")
    return save_path


def export_int8(model, example_input, model_type):
    """Export INT8 quantized model with XNNPACK"""
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
        XNNPACKQuantizer,
    )
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    
    # Pre-autograd export
    pre_autograd = export(model, example_input, strict=True).module()
    
    # Quantization
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(pre_autograd, quantizer)
    
    # Calibration (간단히 example_input만 사용)
    prepared(*example_input)
    
    # Convert
    quantized = convert_pt2e(prepared)
    
    # Export again
    aten_dialect = export(quantized, example_input, strict=True)
    edge_program = to_edge(aten_dialect)
    
    # Delegate to XNNPACK backend
    edge_program = edge_program.to_backend(XnnpackPartitioner())
    
    executorch_program = edge_program.to_executorch(
        ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    
    save_path = os.path.join(RESULTS_DIR, f"{model_type}_executorch_int8.pte")
    with open(save_path, "wb") as f:
        f.write(executorch_program.buffer)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  INT8: {size_mb:.2f} MB")
    return save_path


def export_model(model_type):
    """Export model in all precisions"""
    print(f"\n{model_type.upper()}")
    
    try:
        # Load model
        model = load_model(model_type)
        example_input = (torch.randn(1, 30, 17, 3),)
        
        # FP32
        export_fp32(model, example_input, model_type)
        
        # FP16
        export_fp16(model, example_input, model_type)
        
        return True
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*50)
    print("ExecuTorch Export (FP32/FP16)")
    print("="*50)
    
    models = ['transformer', 'tcn']
    results = {}
    
    for model_type in models:
        success = export_model(model_type)
        results[model_type] = success
    
    print("\n" + "="*50)
    print("Summary:")
    for model_type, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_type}")
    print("="*50)


if __name__ == "__main__":
    main()