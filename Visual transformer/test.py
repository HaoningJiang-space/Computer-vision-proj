import os
import torch
import shutil
import numpy as np
from torch import nn
from model import (vit_base_patch16_224_in21k,
                   vit_base_patch32_224_in21k,
                   vit_large_patch16_224_in21k,
                   vit_large_patch32_224_in21k,
                   vit_huge_patch14_224_in21k)


def set_seed(seed):
    # 设置随机种子

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # 如果使用多GPU训练，给每个GPU都设置种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(args):
    # 创建模型

    if args['model_type'] == "vit_base_patch16_224":
        model = vit_base_patch16_224_in21k(args['num_classes'], has_logits=False)
    elif args['model_type'] == "vit_base_patch32_224":
        model = vit_base_patch32_224_in21k(args['num_classes'], has_logits=False)
    elif args['model_type'] == "vit_large_patch16_224":
        model = vit_large_patch16_224_in21k(args['num_classes'], has_logits=False)
    elif args['model_type'] == "vit_large_patch32_224":
        model = vit_large_patch32_224_in21k(args['num_classes'], has_logits=False)
    elif args['model_type'] == "vit_huge_patch14_224":
        model = vit_huge_patch14_224_in21k(args['num_classes'], has_logits=False)
    else:
        raise Exception("Can't find any model name call {}".format(args['model_type']))

    return model


def model_parallel(args, model):
    # 多GPU并行训练
    device_ids = [i for i in range(len(args['gpu_list'].split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def remove_dir_and_create_dir(dir_name):
    # 将原有文件夹清空
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Created")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Created")
