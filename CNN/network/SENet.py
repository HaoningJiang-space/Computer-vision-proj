import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义SE模块
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SEAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[96, 27, 27]
            SELayer(96),  # 通道注意力机制

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 13, 13]
            SELayer(256),  # 通道注意力机制

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),
            SELayer(384),  # 通道注意力机制

            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),
            SELayer(384),  # 通道注意力机制

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # 256 * [6 * 6] 6x6的图片 / 矩阵
        x = torch.flatten(x, start_dim=1)  # 256 * 6 * 6 数列
        x = self.classifier(x)
        return x

# 创建带有SENet模块的VGGNet实例
# vgg_net = SEVGGNet(5)
# # 打印网络结构
# print(vgg_net)
