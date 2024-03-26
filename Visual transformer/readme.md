# ViT

These code is the implementation of ViT.

将一张图片分成patches；
将patches铺平；
将铺平后的patches的线性映射到更低维的空间；
添加位置embedding编码信息；
将图像序列数据送入标准Transformer encoder中去；
在较大的数据集上预训练；
在下游数据集上微调用于图像分类。

![
](256e7205622f48f0ad6c34471c6ac00f.png)

将图片分类问题转换成了序列问题。即将图片patch转换成 token，以便使用 Transformer 来处理。