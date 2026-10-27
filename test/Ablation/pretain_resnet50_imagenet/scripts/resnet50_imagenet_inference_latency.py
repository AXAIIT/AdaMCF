import os
import sys
import torch
import torchvision.models as models
import numpy as np
import json
from pathlib import Path

# Add project roots to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

@torch.inference_mode()
def benchmark_model(
    model,
    input_shape,
    device="cuda",
    warmup=100,
    iters=200,
    use_amp=True,  # Can set to True for FP16 testing if desired
):
    model = model.to(device)
    model.eval()

    x = torch.randn(*input_shape, device=device)

    # warmup
    for _ in range(warmup):
        if use_amp:
            with torch.amp.autocast(device_type=device if device=="cuda" else "cpu", enabled=True):
                _ = model(x)
        else:
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []

    if device == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            starter.record()
            if use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    _ = model(x)
            else:
                _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        import time
        for _ in range(iters):
            t0 = time.perf_counter()
            if use_amp:
                with torch.amp.autocast(device_type="cpu", enabled=True):
                    _ = model(x)
            else:
                _ = model(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)
    avg_ms = float(times_ms.mean())
    std_ms = float(times_ms.std())
    batch_size = input_shape[0]
    throughput = batch_size / (avg_ms / 1000.0)

    return {
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "throughput": throughput,
        "batch_size": batch_size,
        "input_shape": tuple(input_shape),
    }

def print_latency_tables(model_name, dense_res, pruned_res, variant_name="Pruned"):
    b_N = dense_res['batch_size']
    dense_lt_N = dense_res['avg_ms']
    dense_tp_N = dense_res['throughput']
    pruned_lt_N = pruned_res['avg_ms']
    pruned_tp_N = pruned_res['throughput']
    speedup_N = dense_lt_N / pruned_lt_N if pruned_lt_N > 0 else 0

    print("\n" + "="*85)
    print(f"Latency & Throughput Table (Main Body - Batch={b_N})")
    print("="*85)
    print(f"| {'Model':<12} | {'Variant':<10} | {'Batch':>6} | {'Latency (ms)':>15} | {'Throughput (img/s)':>20} | {'Speedup':>10} |")
    print("-" * 85)
    print(f"| {model_name:<12} | {'Dense':<10} | {b_N:>6} | {dense_lt_N:>15.2f} | {dense_tp_N:>20.2f} | {'1.00x':>10} |")
    print(f"| {model_name:<12} | {variant_name:<10} | {b_N:>6} | {pruned_lt_N:>15.2f} | {pruned_tp_N:>20.2f} | {speedup_N:>9.2f}x |")
    print("="*85)

def print_latency_table_appendix(model_name, dense_res, pruned_res, variant_name="Pruned"):
    dense_lt_1 = dense_res['avg_ms']
    pruned_lt_1 = pruned_res['avg_ms']
    speedup_1 = dense_lt_1 / pruned_lt_1 if pruned_lt_1 > 0 else 0

    print("\n" + "="*65)
    print("Latency Table (Appendix - Batch=1)")
    print("="*65)
    print(f"| {'Model':<12} | {'Variant':<10} | {'Batch=1 Latency (ms)':>22} | {'Speedup':>10} |")
    print("-" * 65)
    print(f"| {model_name:<12} | {'Dense':<10} | {dense_lt_1:>22.2f} | {'1.00x':>10} |")
    print(f"| {model_name:<12} | {variant_name:<10} | {pruned_lt_1:>22.2f} | {speedup_1:>9.2f}x |")
    print("="*65)


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference benchmark on: {device}")

    # Set paths
    base_dir = Path("/workspace/OTOv2_v2/test/Ablation/pretain_resnet50_imagenet")
    dense_model_path = base_dir / "model" / "resnet50_imagenet_torchvision.pth"
    pruned_model_path = base_dir / "cache_resnet50_imagenet_pruned_Adaptive-α_0_5" / "ResNet_compressed.pt"

    # --- 1. Load Baseline (Dense) Model ---
    print(f"Loading dense baseline model from {dense_model_path}...")
    baseline_model = models.resnet50(weights=None)
    state = torch.load(dense_model_path, map_location=device)
    baseline_model.load_state_dict(state)
    baseline_model.to(device)
    baseline_model.eval()

    # --- 2. Load Pruned Subnet Model ---
    if not pruned_model_path.exists():
        print(f"Error: Could not find compressed model at {pruned_model_path}")
        return

    print(f"Loading pruned model from {pruned_model_path}...")
    pruned_model = torch.load(pruned_model_path, map_location=device, weights_only=False)
    
    # Handle DataParallel if it was saved with it
    if isinstance(pruned_model, torch.nn.DataParallel):
        pruned_model = pruned_model.module
        
    pruned_model.to(device)
    pruned_model.eval()

    # --- 3. Run Benchmark ---
    print("\nStarting latency benchmarking (FP32)...")
    
    # Batch = 1
    input_shape_b1 = (1, 3, 224, 224)
    print("Benchmarking Dense [Batch=1]...")
    dense_b1 = benchmark_model(baseline_model, input_shape=input_shape_b1, device=device)
    print("Benchmarking Pruned [Batch=1]...")
    pruned_b1 = benchmark_model(pruned_model, input_shape=input_shape_b1, device=device)

    # Batch = 128
    input_shape_b32 = (128, 3, 224, 224)
    print("Benchmarking Dense [Batch=128]...")
    dense_b32 = benchmark_model(baseline_model, input_shape=input_shape_b32, device=device)
    print("Benchmarking Pruned [Batch=128]...")
    pruned_b32 = benchmark_model(pruned_model, input_shape=input_shape_b32, device=device)

    # --- 4. Print & Save Results ---
    print_latency_tables("ResNet-50", dense_b32, pruned_b32, variant_name="Ours(Sub)")
    print_latency_table_appendix("ResNet-50", dense_b1, pruned_b1, variant_name="Ours(Sub)")

    # Output JSON relative to the cache folder
    output_dir = pruned_model_path.parent
    latency_results = {
        "dense_b1": dense_b1,
        "dense_b32": dense_b32,
        "pruned_b1": pruned_b1,
        "pruned_b32": pruned_b32,
    }
    
    save_path = output_dir / "latency_benchmark.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(latency_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBenchmark results saved successfully to: {save_path}")

if __name__ == "__main__":
    run_benchmark()