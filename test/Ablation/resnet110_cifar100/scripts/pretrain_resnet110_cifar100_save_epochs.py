import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import numpy as np
import random
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.model import resnet110

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_dataloaders(data_root, batch_size=128, num_workers=0):
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

    train_set = CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    test_set = CIFAR100(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader

def check_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total * 100.0

def pretrain_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = '/workspace/data/cifar100'
    train_loader, test_loader = get_dataloaders(data_root)

    model = resnet110(num_classes=100).to(device)
    
    epochs = 120
    milestones = [60, 90]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    save_dir = os.path.join(os.path.dirname(__file__), 'model_snapshots')
    os.makedirs(save_dir, exist_ok=True)
    
    epochs_to_save = [10, 30, 60, 90, 120]
    
    print("\n--- Starting Normal Pretraining for ResNet-110 on CIFAR-100 ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        dt = time.time() - t0
        
        if epoch % 5 == 0 or epoch in epochs_to_save:
            acc = check_accuracy(model, test_loader, device)
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {train_loss:.4f} | Acc: {acc:.2f}% | Time: {dt:.2f}s")
            
        if epoch in epochs_to_save:
            save_path = os.path.join(save_dir, f"chkpt_epoch_{epoch}.pth")
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'acc': acc}, save_path)
            print(f"  --> Saved Checkpoint: {save_path}")

    print("Pretraining Finished!")

if __name__ == "__main__":
    pretrain_model()
