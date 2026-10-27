import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    """
    Kaiming Normal 初始化，适用于 ReLU 激活网络
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        
        # 判断是否需要 Shortcut 投影
        if stride != 1 or in_planes != planes:
            # ------------------------------------------------------------------
            # 【关键修改】始终使用 Option B (1x1 Conv)
            # 原因：结构化剪枝需要 Shortcut 层拥有可学习参数(Weight)，
            # 以便依赖图(Dependency Graph)能正确处理层间通道对齐和剪枝传递。
            # ------------------------------------------------------------------
            if option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
            # Option A (Zero Padding) 不推荐用于剪枝研究，故此处仅作为兼容性保留逻辑，
            # 但不建议实际调用。
            elif option == 'A':
                # 这是一个简单的 Lambda 层实现 Zero Padding，不带参数
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # 初始层：3x3 卷积，适用于 CIFAR 小图
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 堆叠残差块
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # 全连接层
        self.linear = nn.Linear(64, num_classes)

        # 应用权重初始化
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 使用自适应池化，增强鲁棒性 (Output size fixed to 1x1)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==============================================================================
# 模型构建函数
# 默认 num_classes=100 (适配 CIFAR-100)
# ==============================================================================

def resnet20(num_classes=100):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet32(num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)

def resnet44(num_classes=100):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)

def resnet56(num_classes=100):
    """
    ResNet-56 for CIFAR-100
    Total layers: 6 * 9 + 2 = 56
    """
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

def resnet110(num_classes=100):
    """
    ResNet-110 for CIFAR-100
    Total layers: 6 * 18 + 2 = 110
    """
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)

def resnet1202(num_classes=100):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)


# ==============================================================================
# 简单的测试代码 (当直接运行此文件时执行)
# ==============================================================================
if __name__ == "__main__":
    import numpy as np
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Testing ResNet-110 for CIFAR-100...")
    model = resnet110(num_classes=100)
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape} (Expected: [2, 100])")
    

    
   