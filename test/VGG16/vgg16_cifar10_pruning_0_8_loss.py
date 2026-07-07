import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset 
from tqdm.auto import tqdm  
from transformers import get_scheduler  
import numpy as np
import random
import time
from pathlib import Path
from torch.utils import benchmark as torch_benchmark

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

from only_train_once import OTO
from only_train_once.optimizer.alpha_scheduler import ALPHA_CFG, prepare_alpha_eval_batches
from cifar10_dataloader import load_prepared_cifar10_data


def measure_latency_benchmark(model: nn.Module, dummy_input: torch.Tensor, device: torch.device, num_threads: int = 1, num_runs: int = 1000) -> float:
    """
    使用 torch.utils.benchmark 测量单样本推理延迟（ms），最准确、最推荐。
    """
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()

    # 设置线程数（对 CPU 尤其重要）
    torch.set_num_threads(num_threads)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # 同步（GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()

    timer = torch_benchmark.Timer(
        stmt="model(dummy_input)",
        setup="torch.cuda.synchronize()" if device.type == 'cuda' else "",
        globals={"model": model, "dummy_input": dummy_input},
        num_threads=num_threads,
    )

    # 自动运行足够长时间以获得稳定结果
    measurement = timer.blocked_autorange(min_run_time=5)
    
    latency_ms = measurement.median * 1000
    return round(latency_ms, 4)

def check_accuracy(model, loader):
    """
    计算模型在给定数据加载器上的 Top-1 和 Top-5 准确率。
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    top5_num_correct = 0
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in loader:
            x = x.to(model_device, non_blocking=True)
            y = y.to(model_device, non_blocking=True)
            
            scores = model(x)  
            
            # Top-1 准确率
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

            # Top-5 准确率
            _, top5_indices = scores.topk(5, dim=1)
            y_reshaped = y.view(-1, 1)
            top5_correct_batch = (top5_indices == y_reshaped).any(dim=1).sum().item()
            top5_num_correct += top5_correct_batch

    if num_samples == 0:
        print("警告: check_accuracy 接收到的样本数量为0。")
        return 0.0, 0.0

    acc = float(num_correct) / num_samples * 100
    top5_acc = float(top5_num_correct) / num_samples * 100

    return acc, top5_acc

def prune_model(seed=42):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    output_cache_dir = os.path.join(base_dir, 'cache_vgg16_cifar10_pruned')
    os.makedirs(output_cache_dir, exist_ok=True)
    
    batch_size = 256              
    num_workers = 8    

    # --- 第 1 步：加载数据和预训练模型 ---
    print("\n--- 第 1 步：加载数据和预训练模型 ---")

    local_model_path = '/workspace/OTOv2/test/pretrain_vgg16_cifar10/model/VGG16_cifar10.pt'
    local_data_path = '/workspace/data/cifar10'
    model = torch.load(local_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    try:
        trainset = CIFAR10(root=local_data_path, train=True, download=False, transform=transform_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
        print(f"成功加载 CIFAR-10 训练集，包含 {len(trainset)} 张图片。")
        
        testset = CIFAR10(root=local_data_path, train=False, download=False, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
        print(f"成功加载 CIFAR-10 测试集，包含 {len(testset)} 张图片。")
    except Exception as err:
            print(err)
            return

    # --- 第 2 步：初始化 OTO ---
    print("\n--- 第 2 步：初始化 OTO ---")
    # === α-验证子集 ===
    val_batches, alpha_info = prepare_alpha_eval_batches(
            trainset=trainset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    print(
        f"[α-Subset] 来自训练集的 α-更新子集：样本数={alpha_info['n_alpha_val']}, "
        f"batch_size={alpha_info['alpha_batch_size']}"
    )
    print(
        f"α 更新缓存微批：共 {alpha_info['num_val_microbatches']} 个"
        f"（与 ALPHA_CFG.eval_batch_limit 对齐）。"
    )

    dummy_input = torch.rand(1, 3, 32, 32).to(device)  # CIFAR-10 输入尺寸
    oto = OTO(model=model, dummy_input=dummy_input)
    print("正在生成剪枝前的依赖图...")
    oto.visualize(view=False, out_dir=output_cache_dir)

    nodes_to_keep_unpruned = ['node-165']
    
    print(f"尝试标记节点 {nodes_to_keep_unpruned} 为不可剪枝...")
    oto.mark_unprunable_by_node_ids(nodes_to_keep_unpruned)
    print(f"节点 {nodes_to_keep_unpruned} 已标记为不可剪枝。")
    
    # --- 第 3 步：剪枝前评估 ---
    print("\n--- 第 3 步：剪枝前评估 ---")
    
    # 推理延迟（batch=1）
    print("正在测量剪枝前推理延迟 (Batch Size=1)...")
    latency_before = measure_latency_benchmark(model, dummy_input, device, num_threads=1)
    print(f"剪枝前推理延迟 (bs=1): {latency_before} ms")

    print("正在评估剪枝前模型的准确率...")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m = oto.compute_flops(in_million=True)['total']
    params_before_m = oto.compute_num_params(in_million=True)
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    # --- 第 4 步：使用 HESSO 进行剪枝 ---
    print("\n--- 第 4 步：使用 HESSO 进行剪枝 ---")
    print("\n--- 阶段一：预热与剪枝 ---")
    learning_rate = 1e-2             # 阶段一的初始学习率
    warmup_epochs = 10                # 热身阶段周期数
    pruning_epochs = 100             # 剪枝阶段周期数
    pruning_periods = pruning_epochs            
    group_sparsity = 0.8             # 目标组稀疏率
    lr_scheduler_type = "cosine"     
    num_update_steps_per_epoch = len(trainloader)
    pruning_and_warmup_epochs = warmup_epochs + pruning_epochs
    pruning_and_warmup_steps = pruning_and_warmup_epochs * num_update_steps_per_epoch
    
    start_pruning_step = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch

    print(f"目标组稀疏率: {group_sparsity}")
    print(f"剪枝将在第 {start_pruning_step} 步开始，持续 {pruning_steps} 步。")

    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=learning_rate,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        seed = seed,
        )

    lr_scheduler_prune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_prune,
        num_warmup_steps=0,
        num_training_steps=pruning_and_warmup_steps,
    )

    print("\n--- 开始预热与剪枝训练 ---")
    model.train()
    for epoch in range(pruning_and_warmup_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{pruning_and_warmup_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        epoch_start = time.time()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # 同一批样本上，进入新 period 前更新 α
            if optimizer_prune.will_enter_new_period_on_next_step():
                optimizer_prune.alpha_scheduler.update(
                    model=model,
                    criterion=criterion,
                    inputs=val_batches,
                    targets=None,
                    t_in_period=0,
                )
            optimizer_prune.step()
            lr_scheduler_prune.step()
            current_lr = optimizer_prune.param_groups[0]['lr']
            optimizer_prune.zero_grad()
            epoch_loss += loss.item()
            epoch_lr += current_lr
            epoch_steps += 1
        epoch_duration = max(time.time() - epoch_start, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / epoch_duration
        eta = (pruning_and_warmup_epochs - epoch - 1) * epoch_duration
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"[阶段一] Epoch {epoch+1}/{pruning_and_warmup_epochs} 评估 \n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {epoch_duration:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")
        print(dict(zip(optimizer_prune._alpha_keys, optimizer_prune.alpha_scheduler.get_alpha().tolist())))
        metrics = optimizer_prune.compute_metrics()
        print(f"Sparsity: {metrics.group_sparsity:.4f}")    
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")
        model.train()

    print("\n--- 剪枝阶段完成，构建子网络 ---")
    oto.construct_subnet(out_dir=str(output_cache_dir))
    compressed_model_path = oto.compressed_model_path
    
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    compressed_model = model

    latency_after = measure_latency_benchmark(compressed_model, dummy_input, device, num_threads=1)
    speedup = latency_before / latency_after if latency_after > 0 else 0
    print(f"剪枝后推理延迟 (bs=1): {latency_after} ms")
    print(f"实际加速比: {speedup:.2f}x")

    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, testloader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m = oto.compute_flops(in_million=True)['total']
    params_after_m = oto.compute_num_params(in_million=True)
    print(f"剪枝后FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝后参数量: {params_before_m:.2f} M")

    final_metrics = optimizer_prune.compute_metrics(
        flops_before=flops_before_m,
        flops_after=flops_after_m,
        params_before=params_before_m,
        params_after=params_after_m,
    )
    final_group_sparsity = final_metrics.group_sparsity
    pruned_flops_m = final_metrics.pruned_flops
    pruned_params_m = final_metrics.pruned_params
    pruned_flops_percent = final_metrics.pruned_flops_percent
    pruned_params_percent = final_metrics.pruned_params_percent

    print("\n--- 第 6 步：压缩总结 ---")
    print(f"最终剪枝率 (组稀疏率): {final_group_sparsity:.2f} ({final_group_sparsity * 100:.2f}%)")
    print(f"减少的 FLOPs: {pruned_flops_m:.2f}M ({pruned_flops_percent:.2f}%)")
    print(f"减少的参数量: {pruned_params_m:.2f}M ({pruned_params_percent:.2f}%)")

    if final_metrics.flops_before > 0 and final_metrics.flops_after > 0:
        print(f"FLOPs 压缩比: {final_metrics.flops_compression_ratio:.2f}x "
              f"({final_metrics.flops_before:.2f}M -> {final_metrics.flops_after:.2f}M)")
    else:
        print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

    if final_metrics.params_before > 0 and final_metrics.params_after > 0:
        print(f"参数量压缩比: {final_metrics.params_compression_ratio:.2f}x "
              f"({final_metrics.params_before:.2f}M -> {final_metrics.params_after:.2f}M)")
    else:
        print(f"参数量压缩比: N/A (剪枝前: {params_before_m:.2f}M, 剪枝后: {params_after_m:.2f}M)")

    print(f"\n准确率变化: Top-1 从 {accuracy1_before:.2f}% 变为 {accuracy1_after:.2f}% (变化: {accuracy1_after - accuracy1_before:+.2f}%)")
    print(f"准确率变化: Top-5 从 {accuracy5_before:.2f}% 变为 {accuracy5_after:.2f}% (变化: {accuracy5_after - accuracy5_before:+.2f}%)")

    prune_results = {
        "seed": seed,
        "accuracy1_before": accuracy1_before,
        "accuracy5_before": accuracy5_before,
        "accuracy1_after": accuracy1_after,
        "accuracy5_after": accuracy5_after,
        "flops_before_m": flops_before_m,
        "flops_after_m": flops_after_m,
        "params_before_m": params_before_m,
        "params_after_m": params_after_m,
        "final_group_sparsity": final_group_sparsity,
        "pruned_flops_m": pruned_flops_m,
        "pruned_flops_percent": pruned_flops_percent,
        "pruned_params_m": pruned_params_m,
        "pruned_params_percent": pruned_params_percent,
        "flops_compression_ratio": final_metrics.flops_compression_ratio
            if final_metrics.flops_before > 0 and final_metrics.flops_after > 0 else None,
        "params_compression_ratio": final_metrics.params_compression_ratio
            if final_metrics.params_before > 0 and final_metrics.params_after > 0 else None,
        "latency_before": latency_before,
        "latency_after": latency_after,   
        "speedup": speedup,               
        "compressed_model_path": oto.compressed_model_path,
    }
    return prune_results


def finetune_model(seed=42):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    local_data_path = '/workspace/data/cifar10'

    pruned_cache_dir = os.path.join(base_dir, 'cache_vgg16_cifar10_pruned')

    compressed_model_path = None
    if os.path.exists(pruned_cache_dir):
        for f in os.listdir(pruned_cache_dir):
            if f.endswith('_compressed.pt'):
                compressed_model_path = os.path.join(pruned_cache_dir, f)
                break
    
    if not compressed_model_path or not os.path.exists(compressed_model_path):
        print(f"错误：在 {pruned_cache_dir} 未找到压缩模型。请先运行剪枝。")
        return

    print(f"加载压缩后的模型从: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
    
    batch_size = 256
    num_workers = 16
    
    trainloader, testloader, _ , _ = load_prepared_cifar10_data(
        data_root_dir=local_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        download=False,   
    )

    # --- 阶段二：微调 ---
    print("\n--- 阶段二：微调 ---")
    finetune_lr = 3e-3
    num_train_epochs = 300
    warmup_epochs = 5
    total_epochs = num_train_epochs
    num_update_steps_per_epoch = len(trainloader)
    total_steps = total_epochs * num_update_steps_per_epoch
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    lr_scheduler_type = "cosine"

    finetuned_model_dir = os.path.join(base_dir, f'finetuned_models_sparsity_0.8')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    print(f"创建微调模型存储目录: {finetuned_model_dir}")

    best_accuracy1 = 0.0
    best_accuracy5 = 0.0
    best_epoch = 0

    optimizer_finetune = torch.optim.SGD(
        model.parameters(),
        lr=finetune_lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    lr_scheduler_finetune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_finetune,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    criterion = nn.CrossEntropyLoss().to(device)

    print(f"将使用新的标准SGD优化器进行 {warmup_epochs} 个预热周期 + {total_epochs - warmup_epochs} 个微调周期。")
    model.train()
    for epoch in range(total_epochs):
        in_warmup = epoch < warmup_epochs
        phase_tag = "[阶段二-预热]" if in_warmup else "[阶段二-微调]"
        print(f"\n{phase_tag} Epoch {epoch+1}/{total_epochs} 正在训练...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        epoch_start = time.time()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_finetune.step()
            lr_scheduler_finetune.step()
            optimizer_finetune.zero_grad()
            current_lr = optimizer_finetune.param_groups[0]['lr']
            epoch_loss += loss.item()
            epoch_lr += current_lr
            epoch_steps += 1
        epoch_duration = max(time.time() - epoch_start, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / epoch_duration
        eta = (total_epochs - epoch - 1) * epoch_duration
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"{phase_tag} Epoch {epoch+1}/{total_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {epoch_duration:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")

        if accuracy1_epoch > best_accuracy1:
            best_accuracy1 = accuracy1_epoch
            best_epoch = epoch+1
            print(f"当前最佳Top-1准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})")
            finetuned_model_path = os.path.join(finetuned_model_dir, 'vgg16_cifar10_finetuned.pt')
            print(f"  - 保存模型到: {finetuned_model_path}\n")
            torch.save(model, finetuned_model_path)
        elif accuracy1_epoch == best_accuracy1:
            if accuracy5_epoch > best_accuracy5:
                best_accuracy5 = accuracy5_epoch
                print(f"当前最佳Top-1准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})")
                finetuned_model_path = os.path.join(finetuned_model_dir, 'vgg16_cifar10_finetuned.pt')
                print(f"  - 保存模型到: {finetuned_model_path}\n")
                torch.save(model, finetuned_model_path)
        model.train()

    print("微调过程完成。")
    print(f"最佳模型已保存在: {os.path.join(finetuned_model_dir, 'vgg16_cifar10_finetuned.pt')}")
    
    model = torch.load(finetuned_model_path, map_location=device, weights_only=False)
    model.eval()
    accuracy1_after, accuracy5_after = check_accuracy(model, testloader)
    print(f"微调后最终准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    baseline_model_path = '/workspace/OTOv2/test/pretrain_vgg16_cifar10/model/VGG16_cifar10.pt'
    if os.path.exists(baseline_model_path):
        baseline_model = torch.load(baseline_model_path, map_location=device, weights_only=False).to(device)
        baseline_model.eval()
        baseline_top1, baseline_top5 = check_accuracy(baseline_model, testloader)
        print(f"未剪枝模型准确率: Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%")
        print(f"相较未剪枝模型: Top-1 变化 {accuracy1_after - baseline_top1:.2f}%, "
              f"Top-5 变化 {accuracy5_after - baseline_top5:.2f}%")
    else:
        print(f"警告: 未找到未剪枝模型 {baseline_model_path}，无法对比准确率。")

    finetune_results = {
        "seed": seed,
        "finetune_best_epoch": best_epoch,
        "finetune_top1_final": accuracy1_after,
        "finetune_top5_final": accuracy5_after,
        "baseline_top1": baseline_top1 if os.path.exists(baseline_model_path) else None,
        "baseline_top5": baseline_top5 if os.path.exists(baseline_model_path) else None,
        "delta_top1_vs_baseline": (accuracy1_after - baseline_top1) if os.path.exists(baseline_model_path) else None,
        "delta_top5_vs_baseline": (accuracy1_after - baseline_top5) if os.path.exists(baseline_model_path) else None,
        "finetuned_model_path": finetuned_model_path,
    }
    return finetune_results


def main():
    prune_model()
    finetune_model()

if __name__ == "__main__":
    main()
