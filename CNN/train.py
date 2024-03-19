from torchvision import transforms, datasets, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from network.AlexNet import AlexNet
import time



def main():
    # 1. 加载数据集
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                     transforms.ToTensor(),  # 转换为张量
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 标准化
        "val": transforms.Compose([transforms.Resize((224, 224)),  # 重置图像分辨率
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # 重置图像分辨率
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    data_root = "/home/zheng/PycharmProjects/Computer-vision-proj/data_set/flower_data/"  # 数据集路径
    image_datasets = {x: datasets.ImageFolder(root=data_root + x,
                                              transform=data_transform[x])
                      for x in ["train", "val"]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16,
                                 shuffle=True, num_workers=4)
                   for x in ["train", "val"]}

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 定义网络
    model = AlexNet(num_classes=5)
    model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. 训练网络
    epoch = 40
    

            







