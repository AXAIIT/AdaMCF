import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# sys.path.append('/workspace/data')
# from cifar10_dataloader import load_prepared_cifar10_data


current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
sys.path.append(model_dir)

from model import resnet110


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(epoch, net, trainloader, criterion, optimizer, device, total_epochs):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    if epoch == 0:
        print(f'\nEpoch: {epoch} started (Warmup epoch, speed might be slow)...')
    else:
        print(f'\nEpoch: {epoch} started...')

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每 50 个 batch 打印一次
        if (batch_idx + 1) % 50 == 0:
            print(f'   [Epoch {epoch}] Batch {batch_idx + 1}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')

    end_time = time.time()
    epoch_duration = end_time - start_time

    num_batches = len(trainloader)
    it_per_sec = num_batches / epoch_duration if epoch_duration > 0 else 0.0

    # 计算预计剩余时间 (ETA)
    remaining_time = epoch_duration * (total_epochs - epoch - 1)
    rem_hours, rem_rem = divmod(remaining_time, 3600)
    rem_minutes, rem_seconds = divmod(rem_rem, 60)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch: {epoch} Summary | Time: {epoch_duration:.1f}s ({it_per_sec:.2f} it/s) | LR: {current_lr:.6f} | Train Loss: {train_loss/(batch_idx+1):.4f} | Train Acc: {100.*correct/total:.2f}% | ETA: {int(rem_hours)}h {int(rem_minutes)}m {int(rem_seconds)}s')

def test(epoch, net, testloader, criterion, device, best_acc, save_path):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f'Epoch: {epoch} | Test Loss: {test_loss/(batch_idx+1):.4f} | Test Acc: {acc:.2f}%')

    # 保存最佳模型
    if acc > best_acc:
        print(f'Saving best model (Acc: {acc:.2f}%) ...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        torch.save(state, os.path.join(save_path, 'resnet110_best.pth'))
        best_acc = acc
    
    return best_acc

def main():
    # --- 设置随机种子 ---
    set_seed(42)

    # --- 超参数设置 ---
    lr = 0.1
    batch_size = 256 
    num_workers = 8
    epochs = 200
    momentum = 0.9
    weight_decay = 5e-4
    
    # --- 路径设置 ---
    data_root = '/workspace/data/cifar10' 
    save_dir = '/workspace/OTOv2_v2/test/pretrain_resnet110_cifar10/model'
    
    # 设备配置
    device = 'cuda'
    print(f"Using device: {device}")

    # --- 加载数据 ---
    print("Loading Data...")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == 'cuda'))
    print(f"成功加载 CIFAR-10 训练集，包含 {len(trainset)} 张图片。")

    testset = CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda'))
    print(f"成功加载 CIFAR-10 测试集，包含 {len(testset)} 张图片。")


    # --- 构建模型 ---
    print("Building model ResNet110...")
    net = resnet110()
    net = net.to(device)
    
    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print("Using CosineAnnealingLR scheduler.")

    best_acc = 0
    
    # --- 训练循环 ---
    total_start_time = time.time()

    for epoch in range(epochs):
        train(epoch, net, trainloader, criterion, optimizer, device, epochs)
        best_acc = test(epoch, net, testloader, criterion, device, best_acc, save_dir)
        scheduler.step()
        
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Training finished. Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(save_dir, 'resnet110_best.pth')}")

if __name__ == '__main__':
    main()
