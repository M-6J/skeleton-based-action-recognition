# quantization/export_to_tensorrt.py
import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

sys.path.append('/workspace')
from src.dataset import KeypointsDataset
from torch.utils.data import DataLoader
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# ==================== Config ====================
RESULTS_DIR = "/workspace/quantization/results"

BASE_DIR = "/workspace/data/UCF101"
SELECTED_CLASSES = ["Skiing", "PushUps", "Punch", "Biking", "JumpRope",
                    "Diving", "WalkingWithDog", "Rafting", "GolfSwing", "Fencing"]

ONNX_MODELS = {
    'transformer': os.path.join(RESULTS_DIR, 'transformer_fp32_simp.onnx'),
    'tcn': os.path.join(RESULTS_DIR, 'tcn_fp32_simp.onnx')
}

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# ==============================================


class CalibrationDataLoader:
    def __init__(self, num_samples=100, opt_batch_size=8):
        print(f"  Loading calibration data ({num_samples} samples, batch={opt_batch_size})...")
        
        dataset = KeypointsDataset(
            csv_path=os.path.join(BASE_DIR, "train.csv"),
            keypoints_dir=os.path.join(BASE_DIR, "keypoints/train"),
            selected_classes=SELECTED_CLASSES
        )
        
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        self.data = [np.ascontiguousarray(dataset[i][0].numpy(), dtype=np.float32) for i in indices]
        self.opt_batch_size = opt_batch_size
        self.current_index = 0
        
    def get_batch(self):
        # opt_batch_size만큼 묶어서 반환
        if self.current_index >= len(self.data):
            return None
        
        batch_data = []
        for _ in range(self.opt_batch_size):
            if self.current_index < len(self.data):
                batch_data.append(self.data[self.current_index])
                self.current_index += 1
            else:
                break
        
        if len(batch_data) == 0:
            return None
        
        # Pad if not enough samples
        while len(batch_data) < self.opt_batch_size:
            batch_data.append(batch_data[-1])  # Repeat last
        
        # Stack to (opt_batch_size, 30, 17, 3)
        return np.stack(batch_data, axis=0)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.device_input = None
        
    def get_batch_size(self):
        return self.data_loader.opt_batch_size
    
    def get_batch(self, names):
        batch = self.data_loader.get_batch()
        if batch is None:
            return None
        
        # Allocate device memory (첫 호출 시)
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)
        
        try:
            cuda.memcpy_htod(self.device_input, batch)
        except Exception as e:
            print(f"    Calibration batch copy failed: {e}")
            return None
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def build_engine(onnx_path, precision='fp32', calibrator=None):
    """Build TensorRT engine"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(f"  Error parsing ONNX:")
            for error in range(parser.num_errors):
                print(f"    {parser.get_error(error)}")
            return None
    
    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    
    # Precision
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    
    # Optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "keypoints",
        (1, 30, 17, 3),
        (8, 30, 17, 3),
        (16, 30, 17, 3)
    )
    config.add_optimization_profile(profile)
    
    if precision == 'int8' and calibrator:
        config.int8_calibrator = calibrator
        print(f"    Using calibrator for INT8")

    # Build
    print(f"  Building {precision.upper()} engine...")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print(f"  Failed to build {precision.upper()} engine")
            return None
        return serialized_engine
    except Exception as e:
        print(f"  Build failed: {e}")
        return None


def save_engine(serialized_engine, model_name, precision):
    """Save engine file"""
    engine_path = os.path.join(RESULTS_DIR, f"{model_name}_{precision}.engine")
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  {precision.upper()} saved: {size_mb:.2f} MB")
    
    return engine_path, size_mb


def process_model(model_name, onnx_path):
    """Process single model"""
    print(f"\n[{model_name.upper()}] Processing...")
    
    if not os.path.exists(onnx_path):
        print(f"  ONNX not found: {onnx_path}")
        return None
    
    results = {}
    
    try:
        # 1. FP32
        print(f"  [1/3] FP32...")
        engine_fp32 = build_engine(onnx_path, 'fp32')
        if engine_fp32:
            path, size = save_engine(engine_fp32, model_name, 'fp32')
            results['fp32'] = {'path': path, 'size': size}
        
        # 2. FP16
        print(f"  [2/3] FP16...")
        engine_fp16 = build_engine(onnx_path, 'fp16')
        if engine_fp16:
            path, size = save_engine(engine_fp16, model_name, 'fp16')
            results['fp16'] = {'path': path, 'size': size}
        
        # 3. INT8 (TCN only)
        if model_name == 'tcn':
            print(f"  [3/3] INT8...")
            cache_file = os.path.join(RESULTS_DIR, f"{model_name}_calibration.cache")
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            # opt_batch_size=8로 설정!
            data_loader = CalibrationDataLoader(num_samples=100, opt_batch_size=8)
            calibrator = Int8Calibrator(data_loader, cache_file)
            
            engine_int8 = build_engine(onnx_path, 'int8', calibrator)
            if engine_int8:
                path, size = save_engine(engine_int8, model_name, 'int8')
                results['int8'] = {'path': path, 'size': size}
            else:
                print(f"  INT8 build failed, skipping")
        else:
            print(f"  [3/3] INT8 skipped for Transformer")
        
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_summary(all_results):
    """Save summary"""
    summary_path = os.path.join(RESULTS_DIR, "tensorrt_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("TensorRT Export Summary\n")
        f.write("="*70 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n{model_name.upper()}\n")
            f.write("-"*70 + "\n")
            for precision, info in results.items():
                f.write(f"  {precision.upper()}: {info['size']:.2f} MB\n")
    
    print(f"\nSummary saved: {summary_path}")


def main():
    print("="*70)
    print("TensorRT Export (Transformer: FP16, TCN: INT8)")
    print("="*70)
    
    # GPU check
    try:
        cuda.init()
        device = cuda.Device(0)
        print(f"GPU: {device.name()}")
    except Exception as e:
        print(f"Error: GPU not available - {e}")
        return
    
    all_results = {}
    
    for model_name, onnx_path in ONNX_MODELS.items():
        results = process_model(model_name, onnx_path)
        if results:
            all_results[model_name] = results
    
    # Summary
    if all_results:
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        
        for model_name, results in all_results.items():
            print(f"\n{model_name.upper()}:")
            for precision, info in results.items():
                print(f"  {precision.upper()}: {info['size']:.2f} MB")
        
        save_summary(all_results)
        print(f"\nResults saved to: {RESULTS_DIR}")
    else:
        print("\nNo models were exported")
    
if __name__ == "__main__":
    main()