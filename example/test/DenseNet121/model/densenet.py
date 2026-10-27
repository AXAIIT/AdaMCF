"""DenseNet implementation in PyTorch

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """瓶颈层实现(DenseNet-B)
    
    虽然每层只产生 k 个输出特征图，但它通常有更多输入特征图。
    在每个 3×3 卷积前引入 1×1 卷积作为瓶颈层，减少输入特征图数量，
    提高计算效率。
    """
    
    def __init__(self, in_channels, growth_rate):
        """初始化瓶颈层
        
        Args:
            in_channels (int): 输入通道数
            growth_rate (int): 增长率(k)，控制每层输出特征图的数量
        """
        super().__init__()
        # 每个 1×1 卷积产生 4k 个特征图
        inner_channel = 4 * growth_rate

        # DenseNet-B 结构: BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量，连接输入和新生成的特征图
        """
        return torch.cat([x, self.bottle_neck(x)], 1)


class Transition(nn.Module):
    """过渡层实现
    
    过渡层位于密集块之间，执行卷积和池化操作。
    包含批归一化层、1×1卷积层和2×2平均池化层。
    """
    
    def __init__(self, in_channels, out_channels):
        """初始化过渡层
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
        """
        super().__init__()
        # 过渡层结构: BN-Conv(1×1)-AvgPool(2×2)
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 经过降采样后的输出张量
        """
        return self.down_sample(x)


class DenseNet(nn.Module):
    """DenseNet主网络实现
    
    DenseNet-BC版本:
    B: 表示瓶颈层(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
    C: 表示压缩因子(0 <= theta <= 1)
    """
    
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        """初始化DenseNet
        
        Args:
            block (nn.Module): 使用的基本块类型(如Bottleneck)
            nblocks (list): 每个密集块中的层数
            growth_rate (int): 增长率，默认为12
            reduction (float): 压缩因子θ，默认为0.5
            num_class (int): 分类数量，默认为100(CIFAR-100)
        """
        super().__init__()
        self.growth_rate = growth_rate

        # 初始卷积层输出通道数设置为 2 * growth_rate
        inner_channels = 2 * growth_rate

        # 第一个卷积层，使用kernel_size=3(针对CIFAR-100数据集)
        # 对于ImageNet等数据集通常使用kernel_size=7,stride=2
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        # 构建除最后一个密集块外的所有密集块和过渡层
        for index in range(len(nblocks) - 1):
            # 添加密集块
            self.features.add_module(
                "dense_block_layer_{}".format(index), 
                self._make_dense_layers(block, inner_channels, nblocks[index])
            )
            inner_channels += growth_rate * nblocks[index]

            # 添加过渡层，应用压缩因子θ
            out_channels = int(reduction * inner_channels)
            self.features.add_module(
                "transition_layer_{}".format(index), 
                Transition(inner_channels, out_channels)
            )
            inner_channels = out_channels

        # 添加最后一个密集块(不跟过渡层)
        self.features.add_module(
            "dense_block{}".format(len(nblocks) - 1), 
            self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1])
        )
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        
        # 最后添加批归一化和ReLU激活
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        """前向传播过程
        
        Args:
            x (Tensor): 输入图像张量
            
        Returns:
            Tensor: 分类结果
        """
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)  # 展平
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        """构建密集连接层
        
        Args:
            block (nn.Module): 基本块类型
            in_channels (int): 输入通道数
            nblocks (int): 密集块中的层数
            
        Returns:
            nn.Sequential: 构建好的密集块
        """
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module(
                'bottle_neck_layer_{}'.format(index), 
                block(in_channels, self.growth_rate)
            )
            in_channels += self.growth_rate
        return dense_block


def densenet121():
    """创建DenseNet-121模型"""
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    """创建DenseNet-169模型"""
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    """创建DenseNet-201模型"""
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    """创建DenseNet-161模型"""
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)