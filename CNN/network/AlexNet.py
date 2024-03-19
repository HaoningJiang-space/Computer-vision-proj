'''
此部分是关于AlexNet的实现

AlexNet是2012年ImageNet竞赛的冠军，它是第一个使用深度卷积神经网络的模型，它的成功标志着深度学习在计算机视觉领域的应用。
    
AlexNet和LeNet-5的结构类似，但是AlexNet有5个卷积层和3个全连接层，而LeNet-5只有2个卷积层和3个全连接层。

Dropout是AlexNet中的一个重要技术，它可以在训练过程中随机的丢弃一部分神经元，这样可以防止过拟合，提高模型的泛化能力。(随机失活)

Data Augmentation是AlexNet中的另一个重要技术，它可以在训练过程中对图像进行一些随机的变换，比如翻转、缩放、裁剪等，这样可以增加训练数据的多样性，提高模型的泛化能力。

局部响应归一化(Local Response Normalization, LRN)是AlexNet中的另一个重要技术，它可以在训练过程中对神经元的活动进行归一化，这样可以增强模型的泛化能力。

AlexNet的结构如下：

    输入层：227x227x3的图像
    第1个卷积层：96个11x11x3的卷积核，步长为4，激活函数为ReLU
    第1个池化层：3x3的池化核，步长为2
    局部响应归一化
    第2个卷积层：256个5x5x48的卷积核，步长为1，激活函数为ReLU
    第2个池化层：3x3的池化核，步长为2
    局部响应归一化
    第3个卷积层：384个3x3x256的卷积核，步长为1，激活函数为ReLU
    第4个卷积层：384个3x3x192的卷积核，步长为1，激活函数为ReLU
    第5个卷积层：256个3x3x192的卷积核，步长为1，激活函数为ReLU
    第3个池化层：3x3的池化核，步长为2
    全连接层：4096个神经元，激活函数为ReLU
    Dropout：随机丢弃一部分神经元
    全连接层：4096个神经元，激活函数为ReLU
    Dropout：随机丢弃一部分神经元
    输出层：1000个神经元，激活函数为Softmax


'''

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
AlexNet模型
kernel_size: 卷积核大小
stride: 步长
padding: 填充

"""

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(                      #特征提取层
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(96, 256, kernel_size=5, padding=2),   #96个通道输入，256个通道输出,output为256*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          #output为256*13*13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  #output为384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  #output为384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  #output为256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          #output为256*6*6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))         #AdaptiveAvgPool2d(6,6)将任意大小的输入映射到6*6的输出
        self.classifier = nn.Sequential(                    
            nn.Dropout(p=0.5),                              #随机失活
            nn.Linear(256 * 6 * 6, 4096),                   #全连接层,输入维度为256*6*6，输出维度为4096
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                              #随机失活
            nn.Linear(4096, 4096),                          #全连接层,输入维度为4096，输出维度为4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),                   #全连接层,输入维度为4096，输出维度为num_classes
        )

    def forward(self, x):                                   #前向传播
        x = self.features(x)                                #特征提取层
        x = self.avgpool(x)                                 #自适应平均池化 
        x = torch.flatten(x, 1)                             #256*6*6的特征图展平成一维向量
        x = self.classifier(x)
        return x







