from collections import namedtuple
from typing import Optional, Tuple, Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "GoogLeNetOutputs",
    "GoogLeNet",
    "BasicConv2d", "Inception", "InceptionAux",
    "googlenet",
]

# 保持输出结构与原始一致
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}


class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
            self,
            num_classes: int = 100,  # 默认100类（CIFAR-100）
            aux_logits: bool = True,
            transform_input: bool = False,
            dropout: float = 0.3,
            dropout_aux: float = 0.3,
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        # 初始卷积层（适配32x32输入）
        self.conv1 = BasicConv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((3, 3), (1, 1), padding=(1, 1), ceil_mode=True)
        self.conv2 = BasicConv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = BasicConv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((3, 3), (1, 1), padding=(1, 1), ceil_mode=True)

        # Inception模块（修正通道匹配）
        self.inception3a = Inception(256, 80, 120, 160, 20, 40, 40)  # 输出：80+160+40+40=320
        self.inception3b = Inception(320, 160, 160, 240, 40, 120, 80)  # 输出：160+240+120+80=600
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1), ceil_mode=True)

        # 修正：inception4a输入通道=600（来自inception3b输出600）
        self.inception4a = Inception(600, 200, 100, 220, 20, 50, 70)  # 输出：200+220+50+70=540
        # 修正：inception4b输入通道=540（来自inception4a输出540）
        self.inception4b = Inception(540, 160, 112, 224, 24, 64, 64)  # 输出：160+224+64+64=512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)  # 输出：128+256+64+64=512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)  # 输出：112+288+64+64=528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)  # 输出：256+320+128+128=832
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)  # 输出：256+320+128+128=832
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)  # 输出：384+384+128+128=1024

        # 辅助分类器（修正aux1输入通道）
        if aux_logits:
            self.aux1 = InceptionAux(540, num_classes, dropout_aux)  # 输入=inception4a输出540
            self.aux2 = InceptionAux(528, num_classes, dropout_aux)  # 输入=inception4d输出528（正确）
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        if self.training and self.aux_logits:
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        return self._forward_impl(x)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward_impl(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)

        # 前向传播（通道追踪）
        out = self.conv1(x)  # 3→96
        out = self.maxpool1(out)  # 96→96
        out = self.conv2(out)  # 96→96
        out = self.conv3(out)  # 96→256
        out = self.maxpool2(out)  # 256→256

        out = self.inception3a(out)  # 256→320
        out = self.inception3b(out)  # 320→600
        out = self.maxpool3(out)  # 600→600（尺寸32→16）
        out = self.inception4a(out)  # 600→540
        aux1: Optional[Tensor] = None
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(out)

        out = self.inception4b(out)  # 540→512
        out = self.inception4c(out)  # 512→512
        out = self.inception4d(out)  # 512→528
        aux2: Optional[Tensor] = None
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(out)

        out = self.inception4e(out)  # 528→832
        out = self.maxpool4(out)  # 832→832（尺寸16→8）
        out = self.inception5a(out)  # 832→832
        out = self.inception5b(out)  # 832→1024

        out = self.avgpool(out)  # 8x8→1x1
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.fc(out)

        if torch.jit.is_scripting():
            return GoogLeNetOutputs(logits, aux2, aux1)
        else:
            return self.eager_outputs(logits, aux2, aux1)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False,** kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionAux(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(2048, 512)  # 4x4x128=2048
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def googlenet(**kwargs: Any) -> GoogLeNet:
    model = GoogLeNet(** kwargs)
    return model