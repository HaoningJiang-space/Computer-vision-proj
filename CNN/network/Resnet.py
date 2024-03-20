import torch
import torch.nn as nn


# 定义ResNet的基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


# 定义ResNet的主体结构
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # 输入图像大小: (224, 224) -> 输出图像大小: (56, 56)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 输入图像大小: (56, 56) -> 输出图像大小: (28, 28)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 输入图像大小: (28, 28) -> 输出图像大小: (14, 14)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 输入图像大小: (14, 14) -> 输出图像大小: (7, 7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)  # 输入图像大小: (224, 224) -> 输出图像大小: (112, 112)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)  # 输入图像大小: (112, 112) -> 输出图像大小: (56, 56)

        out = self.layer1(out)  # 输入图像大小: (56, 56) -> 输出图像大小: (56, 56)
        out = self.layer2(out)  # 输入图像大小: (56, 56) -> 输出图像大小: (28, 28)
        out = self.layer3(out)  # 输入图像大小: (28, 28) -> 输出图像大小: (14, 14)
        out = self.layer4(out)  # 输入图像大小: (14, 14) -> 输出图像大小: (7, 7)

        out = self.avgpool(out)  # [1]
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# 创建ResNet-50模型
def resnet50(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
