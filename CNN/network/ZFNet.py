import torch
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[96, 27, 27]

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),  # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 13, 13]

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),

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
