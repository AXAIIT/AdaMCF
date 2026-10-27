import sys
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.distributions.beta import Beta


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加模型定义目录（你的GoogLeNet代码所在路径）
MODEL_DEFINITION_ROOT = CURRENT_FILE.parent / "model"  # 替换为你的GoogLeNet代码目录
if str(MODEL_DEFINITION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_DEFINITION_ROOT))

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

model_base_path = os.path.join(project_root, 'model')
if model_base_path not in sys.path:
    sys.path.insert(0, model_base_path)

# 导入CIFAR-100数据加载器（复用你的代码）
from cifar100_dataloader import (
    get_cifar100_train_loader,
    get_cifar100_test_loader,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from googlenet import googlenet


def save_model(model, save_path, epoch, acc1):
    """保存模型权重（包含epoch和准确率，便于后续加载）"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "top1_accuracy": acc1,
    }, save_path)


def load_pretrained_googlenet(weights_path: str, device: torch.device) -> nn.Module:
    """加载训练好的GoogLeNet权重（用于剪枝前的初始化）"""
    model = googlenet(num_classes=100, aux_logits=True)  # CIFAR-100为100类
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model.to(device)


def check_accuracy(model, loader):
    """评估模型Top-1和Top-5准确率"""
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # GoogLeNet测试时仅返回主输出（logits）
            outputs = model(inputs)
            if isinstance(outputs, tuple):  # 确保只取主输出
                outputs = outputs[0]
            _, top1 = outputs.max(dim=1)
            _, top5 = outputs.topk(5, dim=1)
            top1_correct += (top1 == targets).sum().item()
            top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return top1_correct / total * 100.0, top5_correct / total * 100.0


def mixup_data(inputs, targets, alpha=0.4):
    beta_dist = Beta(alpha, alpha)
    lam = beta_dist.sample().item()
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
    targets_a, targets_b = targets, targets[perm]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, preds, targets_a, targets_b, lam):
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)


MIXUP_ALPHA = 0.4
MIXUP_ENABLE_EPOCH = 10          # 预热之后启用
MIXUP_DISABLE_EPOCH = 300        # 300 轮后关闭
AUX_WEIGHT_INIT = 0.3
AUX_DECAY_START = 260            # 260 轮开始衰减
AUX_DECAY_END = 340              # 340 轮衰减到 0


def current_mixup_alpha(epoch: int) -> float:
    if epoch < MIXUP_ENABLE_EPOCH or epoch >= MIXUP_DISABLE_EPOCH:
        return 0.0
    remain = MIXUP_DISABLE_EPOCH - epoch
    window = MIXUP_DISABLE_EPOCH - MIXUP_ENABLE_EPOCH
    return MIXUP_ALPHA * remain / max(1, window)


def current_aux_weight(epoch: int) -> float:
    if epoch < AUX_DECAY_START:
        return AUX_WEIGHT_INIT
    if epoch >= AUX_DECAY_END:
        return 0.0
    ratio = (AUX_DECAY_END - epoch) / max(1, AUX_DECAY_END - AUX_DECAY_START)
    return AUX_WEIGHT_INIT * ratio


def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_dir = CURRENT_FILE.parent
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = output_dir / "best_model.pth"

    batch_size = 256
    num_workers = 36
    data_root_dir = "/workspace/data/cifar100"

    print("加载CIFAR-100数据集...")
    trainloader, trainset = get_cifar100_train_loader(
        data_root_dir=data_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        download=False,
    )
    testloader, testset = get_cifar100_test_loader(
        data_root_dir=data_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        download=False,
    )
    print(f"训练集: {len(trainset)} 张，测试集: {len(testset)} 张，批量 {batch_size}")

    model = googlenet(
        num_classes=100,
        aux_logits=True,
        transform_input=False,
        dropout=0.3,
        dropout_aux=0.3
    ).to(device)
    # 若使用多GPU训练，可添加DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 6. 训练超参数（针对新网络优化）
    learning_rate = 0.1
    num_train_epochs = 300
    warmup_epochs = 10
    num_update_steps_per_epoch = len(trainloader)
    warmup_steps = warmup_epochs * num_update_steps_per_epoch
    total_training_steps = num_train_epochs * num_update_steps_per_epoch

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    scheduler1 = LinearLR(
        optimizer,
        start_factor=0.1,  # 预热起始学习率=0.1*0.08=0.008
        total_iters=warmup_steps  # 预热步数=8轮×每轮步数
    )
    scheduler2 = CosineAnnealingLR(
        optimizer,
        T_max=total_training_steps - warmup_steps,  # 余弦退火步数=总步数-预热步数
        eta_min=1e-6  # 最低学习率
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps]  # 第warmup_steps步后切换到scheduler2
    )

    # 8. 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02).to(device)

    # 9. 训练循环（添加早停策略）
    best_accuracy1 = 0.0
    TARGET_ACC = 78.78
    # patience = 20  # 早停耐心值（连续20轮无提升则终止）
    counter = 0
    progress_bar = tqdm(range(total_training_steps), desc="训练中")

    for epoch in range(num_train_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            mixup_alpha = current_mixup_alpha(epoch)
            use_mixup = mixup_alpha > 0.0
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
            else:
                targets_a = targets_b = None
                lam = 1.0

            # 前向传播：GoogLeNet训练时返回（主输出, aux2, aux1）
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, aux_logits2, aux_logits1 = outputs
            else:
                logits = outputs
                aux_logits2 = aux_logits1 = None

            aux_weight = current_aux_weight(epoch)

            if use_mixup:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                if aux_logits1 is not None and aux_logits2 is not None and aux_weight > 0:
                    loss += aux_weight * mixup_criterion(criterion, aux_logits1, targets_a, targets_b, lam)
                    loss += aux_weight * mixup_criterion(criterion, aux_logits2, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)
                if aux_logits1 is not None and aux_logits2 is not None and aux_weight > 0:
                    loss += aux_weight * criterion(aux_logits1, targets)
                    loss += aux_weight * criterion(aux_logits2, targets)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # 记录训练损失
            train_loss += loss.item() * inputs.size(0)
            progress_bar.update(1)
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_train_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
            )

        # 每轮结束后评估
        model.eval()
        avg_train_loss = train_loss / len(trainset)
        acc1, acc5 = check_accuracy(model, testloader)
        print(
            f"---Epoch {epoch + 1} 测试:",
            f"Epoch {epoch+1} - 训练损失: {avg_train_loss:.4f}, Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%",
            file=sys.stdout,
            flush=True
        )
        model.train()

        # 保存最佳模型与早停判断
        if acc1 > best_accuracy1:
            best_accuracy1 = acc1
            save_model(model, best_weights_path, epoch+1, acc1)
            print(f"保存最佳模型至 {best_weights_path} (Top-1: {best_accuracy1:.2f}%)")
            counter = 0  # 重置计数器
        '''else:
            counter += 1
            print(f"早停计数器: {counter}/{patience}")
            if counter >= patience:
                print(f"早停触发（连续{patience}轮无提升）")
                break  # 终止训练'''
        if acc1 >= TARGET_ACC:
            print(f"达到目标精度 {TARGET_ACC:.2f}%，提前停止训练。")
            break

    print(f"训练完成，最佳Top-1准确率: {best_accuracy1:.2f}%")
    print(f"模型权重已保存至: {best_weights_path}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()