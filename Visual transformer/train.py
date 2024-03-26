import gc
import os
import math
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_data_loader
from utils import remove_dir_and_create_dir, create_model, model_parallel, set_seed


def train(args):
    # 判断训练的硬件为cpu或gpu。
    # 多gpu训练依然需要判断，且需要一个主gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 读取路径
    weights_dir = args['summary_dir'] + "/weights"
    log_dir = args['summary_dir'] + "/logs"

    # 创建/删除路径
    remove_dir_and_create_dir(weights_dir)
    remove_dir_and_create_dir(log_dir)
    writer = SummaryWriter(log_dir)

    # 设置随机种子
    set_seed(10086)

    # 如果硬件够强，可以打开此处代码进行多workers读取数据，否则小心内存爆炸！
    # nw = min([os.cpu_count(), args['batch_size']if args['batch_size'] > 1 else 0, 8])  # number of workers
    # print('使用 {} 个 dataloader workers'.format(nw))

    # 读取数据
    train_loader, train_dataset = get_data_loader(args['train_dir'], args['batch_size'], num_workers=1, aug=True)
    val_loader, val_dataset = get_data_loader(args['val_dir'], args['batch_size'], num_workers=1)
    train_num, val_num = len(train_dataset), len(val_dataset)

    print("使用 {} 张图像作为训练集, {} 张图像作为验证集".format(train_num, val_num))

    # 创建模型
    model = create_model(args)

    # 如果使用预训练模型进行预热，则使用此处代码进行读取
    if args['weights_name'] != "":
        assert os.path.exists(args['weights_name']), "权重文件: '{}' 不存在！".format(args['weights_name'])
        weights_dict = torch.load(args['weights_name'], map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结参数
    if args['use_weights']:
        for name, params in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                params.requires_grad_(False)
            else:
                print("训练 {} 模型".format(name))

    # 使GPU并行运算。如果为单GPU或CPU，代码不会产生效果
    model = model_parallel(args, model)
    model.to(device)

    # 定义loss function。 通常情况下，layer normalization会搭配CrossEntropy使用
    loss_function = torch.nn.CrossEntropyLoss()

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args['lr'], momentum=0.9, weight_decay=5e-5)

    # 设置学习率调整策略。此处可参考文献 https://arxiv.org/pdf/1812.01187.pdf
    # 设置调整策略的lambda
    lf = lambda x: ((1 + math.cos(x * math.pi / args['epochs'])) / 2) * (1 - args['lrf']) + args['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0

    # 开始训练
    for epoch in range(args['epochs']):
        model.train()
        train_acc = 0
        train_loss = []
        train_bar = tqdm(train_loader)
        for data in train_bar:
            train_bar.set_description("epoch {}".format(epoch))
            images, labels = data

            optimizer.zero_grad()

            logits = model(images.to(device))
            prediction = torch.max(logits, dim=1)[1]

            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))

            train_acc += torch.eq(labels.to(device), prediction.to(device)).sum()

            # 清理每一轮训练后的中间变量
            del images, labels
            gc.collect()

        # 验证阶段
        model.eval()
        val_acc = 0
        val_loss = []
        with torch.no_grad():
            for data in val_loader:
                images, labels = data

                logits = model(images.to(device))
                loss = loss_function(logits, labels.to(device))
                prediction = torch.max(logits, dim=1)[1]

                val_loss.append(loss.item())
                val_acc += torch.eq(labels.to(device), prediction.to(device)).sum()

                # 删除每一轮验证后的中间变量
                del images, labels

        val_accurate = val_acc / val_num
        train_accurate = train_acc / train_num
        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), train_accurate, np.mean(val_loss), val_accurate))

        writer.add_scalar("train_loss", np.mean(train_loss), epoch)
        writer.add_scalar("train_acc", train_accurate, epoch)
        writer.add_scalar("val_loss", np.mean(val_loss), epoch)
        writer.add_scalar("val_acc", val_accurate, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), "{}/epoch={}_val_acc={:.4f}.pth".format(weights_dir,
                                                                                   epoch,
                                                                                   val_accurate))

