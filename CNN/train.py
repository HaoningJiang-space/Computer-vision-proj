import torch
import torch.nn as nn
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from network.AlexNet import AlexNet
#from network.ZFNet import ZFNet
import time
import torch.optim as optim


def main():
    batch_size = 128  # 每次进入模型的数据
    num_classes = 5
    net = AlexNet(num_classes=num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('using {} device.'.format(device))
    data_path = os.path.abspath(os.path.join("data"))
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    data_transform = {
        "train": transforms.Compose(
            [transforms.RandomResizedCrop(224),  # 随机裁剪
             transforms.RandomHorizontalFlip(),  # 随机旋转
             transforms.ToTensor(),
             # 把shape=(H x W x C) 的像素值为 [0, 255] 的 PIL.Image 和 numpy.ndarray转换成shape=(C,H,WW)的像素值范围为[0.0, 1.0]的 torch.FloatTensor
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ]
        ),
        "test": transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ]
        )
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    vaild_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["test"])
    train_num = len(train_dataset)
    test_num = len(vaild_dataset)

    n_works = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 9])
    print('Using {} dataloader workers every process'.format(n_works))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_works, drop_last=True)
    valid_loader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=False, num_workers=n_works, drop_last=True)

    print('using {} images for traing and using {} images for testing'.format(train_num, test_num))

    epochs = 40
    save_path = os.path.join(os.getcwd(), 'checkpoints/alex')
    if os.path.isdir(save_path):
        print("checkpoints save in " + save_path)
    else:
        os.makedirs(save_path)
        print("new a dir to save checkpoints: " + save_path)

    best_acc = 0.0
    train_steps = len(train_loader)

    # training
    for epoch in range(epochs):
        time_start = time.time()
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("Epoch" + str(epoch) + ": processing:" + str(step) + "/" + str(train_steps))

        # validate
        time_end = time.time()
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in valid_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / test_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f time_one_epoch: %.3f ' %
              (epoch + 1, running_loss / train_steps, val_accurate, (-time_start + time_end)))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(save_path, 'alex_flower.pth'))

    print('Finished Training')


if __name__ == '__main__':
    main()








