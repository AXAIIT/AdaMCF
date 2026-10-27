import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  
from transformers import get_scheduler  
import numpy as np
import random

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加包含原始 'models' 包的父目录到 sys.path
# 这是 .pt 文件期望 VGG16 类定义所在的模块的根路径
original_model_base_path = os.path.join(project_root, 'test', 'pretrain_vgg16_cifar10', 'CIFAR10_VGG16')
if original_model_base_path not in sys.path:
    sys.path.insert(0, original_model_base_path)
    print(f"Added to sys.path for model loading: {original_model_base_path}")

from only_train_once import OTO

def check_accuracy(model, loader):
    """
    计算模型在给定数据加载器上的 Top-1 和 Top-5 准确率。
    """
    model.eval()
    num_correct = 0
    num_samples = 0
    top5_num_correct = 0

    # 获取模型所在的设备
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

def main():
    # --- 配置 ---
    # 设置随机种子以确保可重复性
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 脚本当前目录: /workspace/UniPruning/test/pretrain_vgg16_cifar10/
    base_dir = os.path.dirname(__file__)
    data_root_dir = "/workspace/data/cifar10"
    output_cache_dir = os.path.join(base_dir, 'cache_vgg16_cifar10_pruned')
    os.makedirs(output_cache_dir, exist_ok=True)
    
    batch_size = 256                 # 保持原有batch size
    num_workers = 48                 # 保持原有workers数

    # --- 第 1 步：加载数据和预训练模型 ---
    print("\n--- 第 1 步：加载数据和预训练模型 ---")

    local_model_path = '/workspace/OTOv2/test/pretrain_vgg16_cifar10/model/VGG16_cifar10.pt'
    model = None

    if os.path.exists(local_model_path):
        print(f"尝试从本地路径 {local_model_path} 加载预训练模型...")
        try:
            loaded_object = torch.load(local_model_path, map_location=device, weights_only=False)
            
            if isinstance(loaded_object, torch.nn.Module):
                model = loaded_object
                print(f"成功从本地路径 {local_model_path} 加载完整的预训练模型对象。")
            else:
                print(f"加载的对象不是 nn.Module (类型: {type(loaded_object)}).")
                from model.VGG16 import VGG16 
                model = VGG16()
                print("将使用位于 'model' 目录下的随机初始化VGG16模型。")

        except ModuleNotFoundError as e_mod:
            print(f"从本地路径加载模型失败 (ModuleNotFoundError): {e_mod}。")
            print("这通常意味着模型文件引用的类定义路径在当前环境中找不到。")
            print("请检查：")
            print(f"  1. `VGG16.py` 是否位于: {os.path.join(original_model_base_path, 'models', 'VGG16.py')}")
            print(f"  2. `{os.path.join(original_model_base_path, 'models')}` 目录下是否有 `__init__.py` 文件。")
            print(f"  3. `{original_model_base_path}` 是否已正确添加到 sys.path。 (当前 sys.path[0]: {sys.path[0]})")
            
            from model.VGG16 import VGG16 
            model = VGG16()
            print("将使用位于 'model' 目录下的随机初始化VGG16模型。")
        except RuntimeError as e_rt:
            print(f"从本地路径加载模型权重失败 (RuntimeError): {e_rt}。")
            from model.VGG16 import VGG16 
            model = VGG16()
            print("将使用位于 'model' 目录下的随机初始化VGG16模型。")
        except Exception as e_other:
            print(f"从本地路径加载模型权重失败 (其他错误): {e_other}。")
            from model.VGG16 import VGG16 
            model = VGG16()
            print("将使用位于 'model' 目录下的随机初始化VGG16模型。")
    else:
        print(f"未找到预训练权重文件: {local_model_path}。")
        from model.VGG16 import VGG16
        model = VGG16()
        print("将使用位于 'model' 目录下的随机初始化VGG16模型。")

    if model is None:
        print("警告: 模型未能成功加载。创建一个新的随机初始化VGG16模型。")
        from model.VGG16 import VGG16
        model = VGG16()
    
    model.to(device)
    model.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        testset = CIFAR10(root=data_root_dir, train=False, download=False, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
        print(f"成功加载 CIFAR-10 测试集，包含 {len(testset)} 张图片。")
        
        # 创建训练数据加载器
        trainset = CIFAR10(root=data_root_dir, train=True, download=False, transform=transform_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
        print(f"成功加载 CIFAR-10 训练集，包含 {len(trainset)} 张图片。")
    except Exception as e:
        print(f"加载 CIFAR-10 数据集失败: {e}")
        print(f"请确保 CIFAR-10 数据位于 '{data_root_dir}' 目录，并且其下有 'cifar-10-batches-py' 子目录。")
        return

    # --- 第 2 步：初始化 OTO ---
    print("\n--- 第 2 步：初始化 OTO ---")
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
    print("正在评估剪枝前模型的准确率 (这可能需要一些时间)...")
    accuracy1_before, accuracy5_before = check_accuracy(model, testloader)
    print(f"剪枝前准确率: Top-1: {accuracy1_before:.2f}%, Top-5: {accuracy5_before:.2f}%")

    flops_before_m = oto.compute_flops(in_million=True)['total']
    params_before_m = oto.compute_num_params(in_million=True)
    print(f"剪枝前 FLOPs: {flops_before_m:.2f} M")
    print(f"剪枝前参数量: {params_before_m:.2f} M")

    original_model_size_bytes = 0
    # 尝试获取本地模型文件的大小
    if local_model_path and os.path.exists(local_model_path):
        original_model_size_bytes = os.stat(local_model_path).st_size
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB")
    else:
        print("无法确定原始模型文件大小，权重文件未找到。")

    # --- 第 4 步：使用 HESSO 进行剪枝和微调 ---
    print("\n--- 第 4 步：使用 HESSO 进行剪枝和微调 ---")

    # 训练阶段参数
    learning_rate = 1e-2             # 阶段一的初始学习率
    finetune_lr = 1e-3               # 阶段二的微调学习率
    num_train_epochs = 300           # 总训练周期
    pruning_epochs = 100             # 剪枝阶段周期数
    pruning_periods = 200            # 剪枝周期数
    group_sparsity = 0.5             # 目标组稀疏率
    lr_scheduler_type = "cosine"     # 学习率调度器类型
    warmup_epochs = 10                # 热身阶段周期数
    
    finetune_epochs = num_train_epochs - warmup_epochs - pruning_epochs
    num_update_steps_per_epoch = len(trainloader)
    finetune_steps = finetune_epochs * num_update_steps_per_epoch

    # --- 阶段一：预热和剪枝 ---
    print("\n--- 阶段一：预热与剪枝 ---")
    pruning_and_warmup_epochs = warmup_epochs + pruning_epochs
    pruning_and_warmup_steps = pruning_and_warmup_epochs * num_update_steps_per_epoch
    
    start_pruning_step = warmup_epochs * num_update_steps_per_epoch
    pruning_steps = pruning_epochs * num_update_steps_per_epoch

    sample_step = 10                                # 每10步采样一次
    min_sample_count = 3                            # 至少需要出现3次才被视为稳定冗余参数

    print(f"目标组稀疏率: {group_sparsity}")
    print(f"总训练周期: {num_train_epochs} = {warmup_epochs}(热身) + {pruning_epochs}(剪枝) + {finetune_epochs}(微调)")
    print(f"剪枝将在第 {start_pruning_step} 步开始，持续 {pruning_steps} 步。")

    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer_prune = oto.hesso(
        variant='sgd',
        lr=learning_rate,
        target_group_sparsity=group_sparsity,
        start_pruning_step=start_pruning_step,
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps,
        # importance_score_criteria = {'l2_magnitude': 0.2, 'l2_avg_magnitude': 0.2,  'cosine_similarity': 0.2, 'taylor_first_order': 0.2, 'taylor_second_order': 0.2}
    )

    lr_scheduler_prune = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer_prune,
        num_warmup_steps=0,
        num_training_steps=pruning_and_warmup_steps,
    )

    print("\n--- 开始预热与剪枝训练 ---")
    progress_bar_prune = tqdm(range(pruning_and_warmup_steps), desc="[阶段一] 剪枝中")
    model.train()
    for epoch in range(pruning_and_warmup_epochs):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_prune.step()
            lr_scheduler_prune.step()
            optimizer_prune.zero_grad()
            progress_bar_prune.update(1)
            metrics = optimizer_prune.compute_metrics()
            current_lr = optimizer_prune.param_groups[0]['lr']
            progress_bar_prune.set_description(
                f"[阶段一] Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                f"Sparsity: {metrics.group_sparsity:.4f}, LR: {current_lr:.6f}"
            )

        print(f"\n--- [阶段一] Epoch {epoch+1}/{pruning_and_warmup_epochs} 评估 ---")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1} 测试准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")
        model.train()

    print("\n--- 剪枝阶段完成，模型结构已固定 ---")

    # --- 构建并加载子网络 ---
    print("\n--- 构建并加载子网络用于微调 ---")
    oto.construct_subnet(out_dir=output_cache_dir)
    compressed_model_path = oto.compressed_model_path
    if not os.path.exists(compressed_model_path):
        print(f"错误：压缩模型未在 {compressed_model_path} 找到。无法进行微调。")
        return
    
    print(f"加载压缩后的模型从: {compressed_model_path}")
    model = torch.load(compressed_model_path, map_location=device, weights_only=False)
    model.to(device)
    print("子网络加载完成，将用此模型进行微调。")

    # --- 阶段二：微调 ---
    print("\n--- 阶段二：微调 ---")
    
    # 添加跟踪最佳准确率的变量
    best_accuracy1 = 0.0
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
        num_warmup_steps=0,
        num_training_steps=finetune_steps,
    )

    print(f"将使用新的标准SGD优化器进行 {finetune_epochs} 个周期的微调。")

    progress_bar_finetune = tqdm(range(finetune_steps), desc="[阶段二] 微调中")
    model.train()
    for epoch in range(finetune_epochs):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_finetune.step()
            lr_scheduler_finetune.step()
            optimizer_finetune.zero_grad()
            progress_bar_finetune.update(1)
            current_lr = optimizer_finetune.param_groups[0]['lr']
            progress_bar_finetune.set_description(
                f"[阶段二] Epoch {epoch+1+pruning_and_warmup_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )
        print(f"\n--- [阶段二] Epoch {epoch+1+pruning_and_warmup_epochs}/{num_train_epochs} 评估 ---")
        model.eval()
        accuracy1_epoch, accuracy5_epoch = check_accuracy(model, testloader)
        print(f"Epoch {epoch+1+pruning_and_warmup_epochs} 测试准确率: Top-1: {accuracy1_epoch:.2f}%, Top-5: {accuracy5_epoch:.2f}%")

        if accuracy1_epoch > best_accuracy1:
            best_accuracy1 = accuracy1_epoch
            best_epoch = epoch+1+pruning_and_warmup_epochs
            print(f"当前最佳Top-1准确率: {best_accuracy1:.2f}% (Epoch {best_epoch})")
            print(f"  - 保存模型到: {compressed_model_path}")
            torch.save(model, compressed_model_path)

        model.train()

    print("剪枝和微调过程完成。")
    print(f"\n--- 保存微调后的模型到: {compressed_model_path} ---")
    print("模型保存成功。")

    model.eval()
    
    print("\n--- 计算最终模型指标 ---")
    final_metrics = optimizer_prune.compute_metrics()
    final_group_sparsity = final_metrics.group_sparsity
    print(f"  - 最终组稀疏度 (Final Group Sparsity): {final_metrics.group_sparsity:.4f}")
    print(f"  - 总可剪枝组数 (Total Prunable Groups): {final_metrics.num_groups}")
    print(f"  - 零权重组数 (Zero Groups): {final_metrics.num_zero_groups}")
    print(f"  - 重要组数 (Important Groups): {final_metrics.num_important_groups}")
    print(f"  - 冗余组数 (Redundant Groups): {final_metrics.num_redundant_groups}")
    print(f"  - 总参数范数 (Total Param Norm): {final_metrics.norm_params:.2f}")

    # --- 第 5 步：构建子网络并进行剪枝后评估 ---
    print("\n--- 第 5 步：最终模型评估 ---")
    
    compressed_model = model
    print("正在评估最终微调后模型的准确率...")
    accuracy1_after, accuracy5_after = check_accuracy(compressed_model, testloader)
    print(f"剪枝后准确率: Top-1: {accuracy1_after:.2f}%, Top-5: {accuracy5_after:.2f}%")

    # 使用 OTO 实例计算剪枝后模型的 FLOPs 和参数 (它内部跟踪了剪枝状态)
    flops_after_m = oto.compute_flops(in_million=True)['total']
    params_after_m = oto.compute_num_params(in_million=True)
    print(f"剪枝后 FLOPs: {flops_after_m:.2f} M")
    print(f"剪枝后参数量: {params_after_m:.2f} M")

    compressed_model_size_bytes = os.stat(oto.compressed_model_path).st_size
    print(f"压缩模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB")

    # --- 第 6 步：压缩总结 ---
    print("\n--- 第 6 步：压缩总结 ---")
    print(f"最终剪枝率 (组稀疏率): {final_group_sparsity:.2f}")
    
    if flops_before_m > 0 and flops_after_m >= 0:
        flops_compression_ratio = flops_before_m / flops_after_m if flops_after_m > 0 else float('inf')
        print(f"FLOPs 压缩比: {flops_compression_ratio:.2f}x ({flops_before_m:.2f}M -> {flops_after_m:.2f}M)")
    else:
        print(f"FLOPs 压缩比: N/A (剪枝前: {flops_before_m:.2f}M, 剪枝后: {flops_after_m:.2f}M)")

    if params_before_m > 0 and params_after_m >= 0:
        params_compression_ratio = params_before_m / params_after_m if params_after_m > 0 else float('inf')
        print(f"参数量压缩比: {params_compression_ratio:.2f}x ({params_before_m:.2f}M -> {params_after_m:.2f}M)")
    else:
        print(f"参数量压缩比: N/A (剪枝前: {params_before_m:.2f}M, 剪枝后: {params_after_m:.2f}M)")

    if original_model_size_bytes > 0 and compressed_model_size_bytes > 0:
        model_size_compression_ratio = original_model_size_bytes / compressed_model_size_bytes
        print(f"模型文件大小压缩比: {model_size_compression_ratio:.2f}x ({original_model_size_bytes / (1024**2):.2f}MB -> {compressed_model_size_bytes / (1024**2):.2f}MB)")
    elif original_model_size_bytes > 0:
        print(f"原始模型文件大小: {original_model_size_bytes / (1024**2):.2f} MB (压缩后模型大小: {compressed_model_size_bytes / (1024**2):.2f}MB)")
    elif compressed_model_size_bytes > 0:
        print(f"压缩后模型文件大小: {compressed_model_size_bytes / (1024**2):.2f} MB (原始模型大小未知)")
    else:
        print("模型文件大小压缩比: N/A (无法获取原始或压缩后的模型大小)")
    
    print(f"\n准确率变化: Top-1 从 {accuracy1_before:.2f}% 变为 {accuracy1_after:.2f}% (变化: {accuracy1_after - accuracy1_before:+.2f}%)")
    print(f"准确率变化: Top-5 从 {accuracy5_before:.2f}% 变为 {accuracy5_after:.2f}% (变化: {accuracy5_after - accuracy5_before:+.2f}%)")

    # --- 第 7 步：导出压缩模型 ---
    print("\n--- 第 7 步：导出压缩模型 ---")
    print(f"压缩后的模型已保存在: {oto.compressed_model_path}")
    print(f"完整的组稀疏模型 (未移除零权重组) 保存在: {oto.full_group_sparse_model_path}")
    print("\n剪枝和压缩测试完成。")

if __name__ == "__main__":
    main()