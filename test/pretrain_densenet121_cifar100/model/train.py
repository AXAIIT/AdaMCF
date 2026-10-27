import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

DATA_MODULE_ROOT = Path("/workspace/data")
if str(DATA_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODULE_ROOT))

from cifar100_dataloader import load_prepared_cifar100_data
from model.densenet import densenet121

def train(epoch, model, trainloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch}: Train Loss: {train_loss/(batch_idx+1):.4f} | Train Acc: {acc:.2f}%")

def validate(epoch, model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch}: Test Loss: {test_loss/(batch_idx+1):.4f}  | Test Acc: {acc:.2f}%")
    return acc

def main():
    # === 配置参数 ===
    lr = 0.1
    epochs = 200
    batch_size = 128
    weight_decay = 5e-4
    momentum = 0.9
    data_root_dir = "/workspace/data/cifar100"
    save_dir = '/workspace/OTOv2_v2/test/pretrain_densenet121_cifar100/model'
    num_workers = 8
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # === 1. 数据准备 ===
    print("加载 CIFAR-100 数据集...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]),
    ])

    trainloader, testloader, trainset, testset = load_prepared_cifar100_data(
        data_root_dir=data_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device == 'cuda',
        download=False,
        train_transform=transform_train,
        test_transform=transform_test,
    )
    print(
        f"训练集: {len(trainset)} 张，测试集: {len(testset)} 张，批量 {batch_size}"
    )

    # === 2. 模型构建 ===
    print('==> Building model..')
    model = densenet121()
    model = model.to(device)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # === 3. 优化器与损失函数 ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # === 4. 训练循环 ===
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        
        train(epoch, model, trainloader, criterion, optimizer, device)
        acc = validate(epoch, model, testloader, criterion, device)
        
        scheduler.step()
        
        # 保存最佳模型
        if acc > best_acc:
            print(f'Saving best model.. (Acc: {acc:.2f}%)')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(save_dir, 'densenet121_cifar100_best.pth'))
            best_acc = acc
        
        # 定期保存 checkpoint
        if (epoch + 1) % 50 == 0:
             torch.save(model.state_dict(), os.path.join(save_dir, f'densenet121_epoch_{epoch+1}.pth'))

        print(f"Time per epoch: {time.time() - start_time:.2f}s\n")

    print(f"Training finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
