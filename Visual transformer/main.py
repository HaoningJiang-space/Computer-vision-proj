import os

import torch
from PIL import Image

from dataloader import data_transform
from utils import create_model, model_parallel


def predict(args):
    # 判断训练的硬件为cpu或gpu。
    # 多gpu训练依然需要判断，且需要一个主gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 读取要预测的图片
    image_path = args['predicted_image']
    assert os.path.exists(image_path), "文件: '{}' 不存在！".format(image_path)
    image = Image.open(image_path)

    # 图像增强
    image = data_transform["val"](image)
    # 图像打平
    image = torch.unsqueeze(image, dim=0)

    # 准备模型。此处创建的模型需要与保存参数文件的模型为同一种模型！
    model = create_model(args)
    model = model_parallel(args, model).to(device)
    # 读取训练好的模型的参数
    model_weight_path = args['saved_pth']
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        index = torch.argmax(predict).numpy()

    print("该内容被预测为: {}   。其概率为: {:.3}\n".format(args['label_name'][index],
                                                predict[index].numpy()))
    for i in range(len(predict)):
        print("预测为: {}   的概率: {:.3}".format(args['label_name'][i],
                                               predict[i].numpy()))




