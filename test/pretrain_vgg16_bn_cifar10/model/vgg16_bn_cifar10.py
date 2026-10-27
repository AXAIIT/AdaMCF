import torch
import torch.nn as nn
from typing import List, Union, cast


class VGG16_BN_CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(VGG16_BN_CIFAR10, self).__init__()
        
        # VGG16 特征提取层
        self.features = self._make_layers()
        
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )

    def _make_layers(self) -> nn.Sequential:
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers: List[nn.Module] = []
        in_channels = 3
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def load_checkpoint(self, checkpoint_path: str, device='cuda', strict=True):
        """加载检查点并移到GPU"""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        self.load_state_dict(state_dict, strict=strict)
        self.to(device)
        
        return checkpoint.get('acc', None)


def create_model(num_classes=10, checkpoint_path=None):
    """创建模型并自动使用GPU"""
    model = VGG16_BN_CIFAR10(num_classes=num_classes)
    
    if checkpoint_path:
        accuracy = model.load_checkpoint(checkpoint_path)
        return model, accuracy
    else:
        model.to('cuda')
        return model
    
    