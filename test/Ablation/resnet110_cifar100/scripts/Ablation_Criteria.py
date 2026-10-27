import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
import random
from thop import profile
from pathlib import Path
import time
import matplotlib.pyplot as plt
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from only_train_once import OTO
from model.model import resnet110 

def check_accuracy(model, loader):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, top1 = outputs.max(dim=1)
            _, top5 = outputs.topk(5, dim=1)
            top1_correct += (top1 == targets).sum().item()
            top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

    if total == 0:
        return 0.0, 0.0
    return top1_correct / total * 100.0, top5_correct / total * 100.0

def compute_flops_with_thop(model, input_size=(1, 3, 32, 32)):
    import copy
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)
    flops_m = macs * 2 / 1e6
    params_m = params / 1e6
    return flops_m, params_m

def run_pruning_experiment(exp_name, importance_score_criteria, seed=42):
    """运行单个剪枝实验"""
    base_dir = os.path.dirname(__file__)
    data_root = '/workspace/data/cifar100'
    pretrained_ckpt_path = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model/best_model.pth'
    output_cache_dir = os.path.join(base_dir, f'cache_resnet110_cifar100_pruned_{exp_name}')
    os.makedirs(output_cache_dir, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 运行实验: {exp_name} ===")
    print(f"Device: {device}")
    print(f"Importance score criteria: {importance_score_criteria}")

    # 数据加载
    batch_size = 128
    num_workers = 16

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
    test_set = CIFAR100(root=data_root, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Loading pretrained weights from: {pretrained_ckpt_path}")
    model = resnet110(num_classes=100)
    
    ckpt = torch.load(pretrained_ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    print("\n--- 剪枝前评估 ---")
    accuracy1_before, accuracy5_before = check_accuracy(model, test_loader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m, params_before_m = compute_flops_with_thop(model, input_size=(1, 3, 32, 32))
    print(f"剪枝前 FLOPs(thop): {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)

    # 配置 HESSO 优化器
    lr = 0.01
    target_sparsity = 0.5
    
    warmup_epochs = 5
    pruning_epochs = 100
    total_epochs = warmup_epochs + pruning_epochs
    
    steps_per_epoch = len(train_loader)
    start_pruning_step = warmup_epochs * steps_per_epoch
    pruning_steps = pruning_epochs * steps_per_epoch

    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=lr,
        first_momentum=0.9,
        second_momentum=0.0,
        dampening=0.0,
        weight_decay=1e-4,
        target_group_sparsity=target_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_epochs,
        pruning_steps=pruning_steps,
        device=device,
        importance_score_criteria=importance_score_criteria  # 使用指定的权重配置
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_prune, T_max=total_epochs)

    print("\n--- 阶段一：预热与剪枝 ---")
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    
    for epoch in range(total_epochs):
        print(f"\n[阶段一] Epoch {epoch+1}/{total_epochs} 正在剪枝...")
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_prune.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_prune.step()
            cur_lr = optimizer_prune.param_groups[0]['lr']
            epoch_loss += loss.item()
            epoch_lr += cur_lr
            epoch_steps += 1
        
        scheduler.step()
        
        
        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (total_epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        print(f"[阶段一] Epoch {epoch+1}/{total_epochs} 评估\n"
              f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
              f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
              f"估计剩余: {eta_h}h:{eta_m}m")

        metrics = optimizer_prune.compute_metrics()
        print(f"Sparsity: {metrics.group_sparsity:.4f}")
        print(dict(zip(optimizer_prune._alpha_keys,
                       optimizer_prune.alpha_scheduler.get_alpha().tolist())))

        model.eval()
        acc1_ep, acc5_ep = check_accuracy(model, test_loader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {acc1_ep:.2f}%, Top-5: {acc5_ep:.2f}%")
        model.train()

    print("\n--- 构建并加载子网络用于微调 ---")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    compressed_model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    compressed_model.to(device)
    compressed_model.eval()

    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, test_loader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    flops_after_m, params_after_m = compute_flops_with_thop(compressed_model, input_size=(1, 3, 32, 32))
    print(f"剪枝后 FLOPs(thop): {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

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

    # 返回实验结果
    result = {
        'exp_name': exp_name,
        'importance_score_criteria': importance_score_criteria,
        'accuracy1_before': accuracy1_before,
        'accuracy5_before': accuracy5_before,
        'accuracy1_after': accuracy1_after,
        'accuracy5_after': accuracy5_after,
        'flops_before': flops_before_m,
        'flops_after': flops_after_m,
        'params_before': params_before_m,
        'params_after': params_after_m,
        'sparsity': final_group_sparsity,
        'pruned_flops_percent': pruned_flops_percent,
        'pruned_params_percent': pruned_params_percent,
        'accuracy_drop': accuracy1_before - accuracy1_after,
        'compressed_model_path': compressed_model_path
    }
    
    # 保存实验结果
    result_path = os.path.join(output_cache_dir, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def finetune_model(compressed_model_path, exp_name, seed=42):
    """对剪枝后的模型进行微调"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 微调实验: {exp_name} ===")
    print(f"使用设备: {device}")

    base_dir = os.path.dirname(__file__)
    local_data_path = "/workspace/data/cifar100"

    finetuned_model_dir = os.path.join(base_dir, f'finetuned_{exp_name}')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = os.path.join(finetuned_model_dir, f'resnet110_cifar100_finetuned_{exp_name}.pt')

    if not os.path.exists(compressed_model_path):
        print(f"错误：未找到压缩模型 {compressed_model_path}，请先运行剪枝阶段。")
        return None

    print(f"加载压缩模型: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)

    batch_size = 128
    num_workers = 16

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = CIFAR100(root=local_data_path, train=True, download=False, transform=transform_train)
    test_set = CIFAR100(root=local_data_path, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    epochs = 300
    lr = 0.01
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_lr = 0.0
        epoch_steps = 0
        t0 = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        epoch_loss += loss.item()
        cur_lr = optimizer.param_groups[0]['lr']
        epoch_loss += loss.item()
        epoch_lr += cur_lr
        epoch_steps += 1

        dt = max(time.time() - t0, 1e-8)
        avg_loss = epoch_loss / epoch_steps
        avg_lr = epoch_lr / epoch_steps
        it_per_sec = epoch_steps / dt
        eta = (epochs - epoch - 1) * dt
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        
        if epoch % 10 == 0 or epoch == epochs - 1:  # 每10个epoch或最后一个epoch评估
            model.eval()
            accuracy1_after, accuracy5_after = check_accuracy(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs} 评估\n"
                  f"平均Loss: {avg_loss:.4f}, 平均LR: {avg_lr:.6f}, "
                  f"吞吐: {it_per_sec:.2f} it/s, 用时: {dt:.2f}s, "
                  f"估计剩余: {eta_h}h:{eta_m}m")
            print(f"Epoch {epoch+1} 测试准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

            if accuracy1_after > best_top1 or (accuracy1_after == best_top1 and accuracy5_after > best_top5):
                best_top1 = accuracy1_after
                best_top5 = accuracy5_after
                best_epoch = epoch + 1
                print(f"当前最佳Top-1准确率: {best_top1:.2f}% (Epoch {best_epoch})")
                print(f"  - 保存模型到: {finetuned_model_path}")
                torch.save(model, finetuned_model_path)

        model.train()

    print(f"微调过程完成。实验: {exp_name}")
    print(f"最佳模型已保存在: {finetuned_model_path}")

    model = torch.load(finetuned_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    top1_final, top5_final = check_accuracy(model, test_loader)
    print(f"微调后最终准确率: Top-1: {top1_final:.2f}%, Top-5: {top5_final:.2f}%")

    return {
        'finetuned_model_path': finetuned_model_path,
        'finetuned_top1_accuracy': top1_final,
        'finetuned_top5_accuracy': top5_final,
        'best_epoch': best_epoch
    }

def create_comparison_plot(results, output_dir):
    """创建柱状图比较实验结果"""
    exp_names = [r['exp_name'] for r in results]
    accuracy_drops = [r['accuracy_drop'] for r in results]
    sparsity = [r['sparsity'] * 100 for r in results]  # 转换为百分比
    flops_reduction = [r['pruned_flops_percent'] for r in results]
    params_reduction = [r['pruned_params_percent'] for r in results]
    accuracy_after = [r['accuracy1_after'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 准确率下降柱状图
    axes[0, 0].bar(exp_names, accuracy_drops, color='lightcoral')
    axes[0, 0].set_title('Accuracy Drop After Pruning')
    axes[0, 0].set_ylabel('Accuracy Drop (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 剪枝率柱状图
    axes[0, 1].bar(exp_names, sparsity, color='lightblue')
    axes[0, 1].set_title('Group Sparsity After Pruning')
    axes[0, 1].set_ylabel('Sparsity (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # FLOPs 减少百分比柱状图
    axes[1, 0].bar(exp_names, flops_reduction, color='lightgreen')
    axes[1, 0].set_title('FLOPs Reduction Percentage')
    axes[1, 0].set_ylabel('FLOPs Reduction (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 参数减少百分比柱状图
    axes[1, 1].bar(exp_names, params_reduction, color='lightsalmon')
    axes[1, 1].set_title('Parameters Reduction Percentage')
    axes[1, 1].set_ylabel('Parameters Reduction (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'resnet110_cifar100_ablation_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"比较图已保存至: {plot_path}")

def create_detailed_comparison_plot(results, output_dir):
    """创建更详细的比较图"""
    exp_names = [r['exp_name'] for r in results]
    accuracy_before = [r['accuracy1_before'] for r in results]
    accuracy_after = [r['accuracy1_after'] for r in results]
    accuracy_drops = [r['accuracy_drop'] for r in results]
    
    x = np.arange(len(exp_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 准确率对比柱状图
    ax.bar(x - width/2, accuracy_before, width, label='Before Pruning', alpha=0.8)
    ax.bar(x + width/2, accuracy_after, width, label='After Pruning', alpha=0.8)
    
    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy Comparison Across Different Importance Score Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45)
    ax.legend()
    
    # 添加数值标签
    for i, (before, after) in enumerate(zip(accuracy_before, accuracy_after)):
        ax.text(i - width/2, before + 0.2, f'{before:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, after + 0.2, f'{after:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'resnet110_cifar100_accuracy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"详细比较图已保存至: {plot_path}")

def save_experiment_log(results, output_dir):
    """保存实验日志到txt文件"""
    log_path = os.path.join(output_dir, 'experiment_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== ResNet-110 CIFAR-100 Ablation Study on Importance Score Criteria ===\n\n")
        
        for r in results:
            f.write(f"实验名称: {r['exp_name']}\n")
            f.write(f"重要性分数配置: {r['importance_score_criteria']}\n")
            f.write(f"剪枝前准确率: Top-1: {r['accuracy1_before']:.2f}%, Top-5: {r['accuracy5_before']:.2f}%\n")
            f.write(f"剪枝后准确率: Top-1: {r['accuracy1_after']:.2f}%, Top-5: {r['accuracy5_after']:.2f}%\n")
            f.write(f"准确率下降: {r['accuracy_drop']:.2f}%\n")
            f.write(f"剪枝率: {r['sparsity']:.2f}%\n")
            f.write(f"FLOPs 减少: {r['pruned_flops_percent']:.2f}%\n")
            f.write(f"参数减少: {r['pruned_params_percent']:.2f}%\n")
            f.write(f"压缩模型路径: {r['compressed_model_path']}\n")
            f.write("-" * 80 + "\n")
        
        # 写入汇总统计
        f.write("\n=== 汇总统计 ===\n")
        avg_accuracy_drop = np.mean([r['accuracy_drop'] for r in results])
        avg_sparsity = np.mean([r['sparsity'] for r in results])
        avg_flops_reduction = np.mean([r['pruned_flops_percent'] for r in results])
        avg_params_reduction = np.mean([r['pruned_params_percent'] for r in results])
        
        f.write(f"平均准确率下降: {avg_accuracy_drop:.2f}%\n")
        f.write(f"平均剪枝率: {avg_sparsity:.2f}%\n")
        f.write(f"平均FLOPs减少: {avg_flops_reduction:.2f}%\n")
        f.write(f"平均参数减少: {avg_params_reduction:.2f}%\n")
        
        # 找到最佳配置（准确率下降最小）
        best_config = min(results, key=lambda x: x['accuracy_drop'])
        f.write(f"\n最佳配置（准确率下降最小）: {best_config['exp_name']}\n")
        f.write(f"准确率下降: {best_config['accuracy_drop']:.2f}%\n")
        f.write(f"配置: {best_config['importance_score_criteria']}\n")
    
    print(f"实验日志已保存至: {log_path}")

def main():
    # 定义不同重要性分数权重配置的实验
    experiments = [
        {
            'name': 'Uniform_Weights',
            'criteria': {
                'l1_avg_magnitude': 0.2, 
                'l2_avg_magnitude': 0.2, 
                'cosine_similarity': 0.2, 
                'fisher': 0.2, 
                'grad_weight_product': 0.2
            }
        },
        {
            'name': 'Weights_00',
            'criteria': {
                'l1_avg_magnitude': 0.1, 
                'l2_avg_magnitude': 0.1, 
                'cosine_similarity': 0.2, 
                'fisher': 0.2, 
                'grad_weight_product': 0.4
            }
        },
        {
            'name': 'Weights_01',
            'criteria': {
                'l1_avg_magnitude': 0.0, 
                'l2_avg_magnitude': 0.1, 
                'cosine_similarity': 0.3, 
                'fisher': 0.3, 
                'grad_weight_product': 0.3
            }
        },
        {
            'name': 'L1_Biased',
            'criteria': {
                'l1_avg_magnitude': 0.5, 
                'l2_avg_magnitude': 0.1, 
                'cosine_similarity': 0.1, 
                'fisher': 0.1, 
                'grad_weight_product': 0.2
            }
        },
        {
            'name': 'L2_Biased',
            'criteria': {
                'l1_avg_magnitude': 0.1, 
                'l2_avg_magnitude': 0.5, 
                'cosine_similarity': 0.1, 
                'fisher': 0.1, 
                'grad_weight_product': 0.2
            }
        },
        {
            'name': 'Fisher_Biased',
            'criteria': {
                'l1_avg_magnitude': 0.1, 
                'l2_avg_magnitude': 0.1, 
                'cosine_similarity': 0.1, 
                'fisher': 0.6, 
                'grad_weight_product': 0.1
            }
        },
        {
            'name': 'GradWeight_Biased',
            'criteria': {
                'l1_avg_magnitude': 0.1, 
                'l2_avg_magnitude': 0.1, 
                'cosine_similarity': 0.1, 
                'fisher': 0.1, 
                'grad_weight_product': 0.6
            }
        },
        {
            'name': 'Cosine_Biased',
            'criteria': {
                'l1_avg_magnitude': 0.1, 
                'l2_avg_magnitude': 0.1, 
                'cosine_similarity': 0.6, 
                'fisher': 0.1, 
                'grad_weight_product': 0.1
            }
        }
    ]
    
    results = []
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'ablation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行所有剪枝实验
    for exp in experiments:
        result = run_pruning_experiment(exp['name'], exp['criteria'])
        results.append(result)
    
    # 对每个剪枝后的模型进行微调
    finetune_results = []
    for result in results:
        exp_name = result['exp_name']
        compressed_model_path = result['compressed_model_path']
        finetune_result = finetune_model(compressed_model_path, exp_name)
        if finetune_result:
            finetune_results.append({
                'exp_name': exp_name,
                **finetune_result
            })
            # 更新原始结果，添加微调后的准确率
            for r in results:
                if r['exp_name'] == exp_name:
                    r['finetuned_top1_accuracy'] = finetune_result['finetuned_top1_accuracy']
                    r['finetuned_top5_accuracy'] = finetune_result['finetuned_top5_accuracy']
    
    # 创建比较图
    create_comparison_plot(results, output_dir)
    create_detailed_comparison_plot(results, output_dir)
    
    # 保存实验日志
    save_experiment_log(results, output_dir)
    
    # 打印汇总结果
    print("\n=== 汇总结果 ===")
    for r in results:
        print(f"实验: {r['exp_name']}")
        print(f"  重要性分数配置: {r['importance_score_criteria']}")
        print(f"  剪枝前准确率: {r['accuracy1_before']:.2f}%")
        print(f"  剪枝后准确率: {r['accuracy1_after']:.2f}%")
        if 'finetuned_top1_accuracy' in r:
            print(f"  微调后准确率: {r['finetuned_top1_accuracy']:.2f}%")
        print(f"  准确率下降: {r['accuracy_drop']:.2f}%")
        print(f"  剪枝率: {r['sparsity']:.2f}%")
        print(f"  FLOPs 减少: {r['pruned_flops_percent']:.2f}%")
        print(f"  参数减少: {r['pruned_params_percent']:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()