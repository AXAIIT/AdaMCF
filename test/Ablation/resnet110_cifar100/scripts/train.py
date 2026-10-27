import os
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from model.model import resnet110


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def get_loaders(data_root, batch_size=128, num_workers=8):
    # CIFAR-100 标准均值方差
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        total += targets.size(0)
        total_correct += preds.eq(targets).sum().item()
        total_loss += loss.item() * targets.size(0)

    return total_loss / total, 100.0 * total_correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            _, preds = outputs.max(1)
            total += targets.size(0)
            total_correct += preds.eq(targets).sum().item()
            total_loss += loss.item() * targets.size(0)

    return total_loss / total, 100.0 * total_correct / total

def save_checkpoint(state, is_best, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 只保存最新的和最好的，节省空间
    torch.save(state, os.path.join(out_dir, "last_checkpoint.pth"))
    if is_best:
        torch.save(state, os.path.join(out_dir, "best_model.pth"))

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- Settings ---
    data_root = "/workspace/data/cifar100"
    out_dir = "/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model"
    
    epochs = 200
    batch_size = 128
    base_lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    seed = 42
    
    # --- Setup ---
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 日志记录器
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
    
    train_loader, test_loader = get_loaders(data_root, batch_size)
    
    model = resnet110(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 标准 SGD 设置
    optimizer = optim.SGD(model.parameters(), 
                          lr=base_lr, 
                          momentum=momentum, 
                          weight_decay=weight_decay, 
                          nesterov=True)
    
    # 标准 Cosine Annealing 调度器 (按 Epoch 更新)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}] LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Scheduler Update (End of Epoch)
        scheduler.step()
        
        # Logging
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")
        
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, out_dir)

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/3600:.2f} hours.")
    print(f"Best Top-1 Accuracy: {best_acc:.2f}%")
    writer.close()

if __name__ == "__main__":
    main()
