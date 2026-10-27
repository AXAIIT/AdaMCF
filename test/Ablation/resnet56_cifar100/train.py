import os
import time
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from transformers import get_scheduler


from model.model import resnet56


def get_loaders(data_root, batch_size=128, num_workers=4, pin_memory=True, download=True):
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

    train_set = CIFAR100(root=data_root, train=True, download=download, transform=transform_train)
    test_set = CIFAR100(root=data_root, train=False, download=download, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        _, preds = outputs.max(1)
        total += targets.size(0)
        total_correct += preds.eq(targets).sum().item()
        total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * total_correct / total
    return avg_loss, acc


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

    avg_loss = total_loss / total
    acc = 100.0 * total_correct / total
    return avg_loss, acc


def save_checkpoint(state, is_best, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pth")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pth")
        torch.save(state, best_path)


def main():
    # Training parameters matching the densenet approach
    data_root = "/workspace/data/cifar100"
    epochs = 200
    warmup_epochs = 10
    batch_size = 128
    base_lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 16
    out_dir = "/workspace/OTOv2_v2/test/Ablation/resnet56_cifar100/model"
    seed = 42

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = resnet56().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Create optimizer using the same approach as in densenet fine-tuning
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Create learning rate scheduler using the same approach as in densenet fine-tuning
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_scheduler_type = "cosine"
    
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch = 0
    best_acc = 0.0

    for epoch in range(start_epoch, epochs):
        in_warmup = epoch < warmup_epochs
        phase = "[Warmup]" if in_warmup else "[Training]"
        print(f"\n{phase} Epoch [{epoch+1}/{epochs}] Training...")
        
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, lr_scheduler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                "best_acc": best_acc
            },
            is_best=is_best,
            out_dir=out_dir,
        )

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"{phase} Epoch [{epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% | "
            f"best={best_acc:.2f}% | lr={current_lr:.6f} | time={dt:.1f}s"
        )

    print(f"Training finished. Best test acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()


'''
import os
import time
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms


from model.model import resnet56


def get_loaders(data_root, batch_size=128, num_workers=4, pin_memory=True, download=True):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = CIFAR100(root=data_root, train=True, download=download, transform=transform_train)
    test_set = CIFAR100(root=data_root, train=False, download=download, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
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

    avg_loss = total_loss / total
    acc = 100.0 * total_correct / total
    return avg_loss, acc


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

    avg_loss = total_loss / total
    acc = 100.0 * total_correct / total
    return avg_loss, acc


def save_checkpoint(state, is_best, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pth")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pth")
        torch.save(state, best_path)


def main():
    data_root = "/workspace/data/cifar100"
    epochs = 200
    warmup_epochs = 5
    batch_size = 128
    base_lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 16
    out_dir = "/workspace/OTOv2_v2/test/Ablation/resnet56_cifar100/model"
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = resnet56().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Cosine 退火 + 线性 warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # epoch 从 0 开始
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_acc = 0.0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc
            },
            is_best=is_best,
            out_dir=out_dir,
        )

        dt = time.time() - t0
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% | "
            f"best={best_acc:.2f}% | time={dt:.1f}s"
        )

    print(f"Training finished. Best test acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
'''