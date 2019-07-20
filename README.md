# PaddleSolution

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/jiangjiajun/PaddleSolution) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle提供了针对视觉任务的端到端企业解决方案，覆盖了从数据准备到模型部署的整个流程。在此Repo中，针对每种视觉任务，我们提供了现阶段精度和效率皆优的神经网络模型，并展示了如何准备数据，如何使用PaddlePaddle完成模型的训练、压缩和部署。


## 目录
* [1 简介](#1-简介)
* [2 数据准备]()
* [3 模型训练](./docs/3_模型训练/3_模型训练.md)
  * 3.1 安装
  * 3.2 数据准备
  * 3.3 训练
  * 3.4 评估
  * 3.5 预测
* [4 模型调优](./docs/4_模型调优/4_模型调优.md)

## 1 简介

PaddleSolution的目标是通过提供针对视觉任务的端到端解决方案，帮助用户打通从准备数据，训练模型，到压缩和部署模型的全部流程。目前支持的视觉任务有以下两种：

* 目标检测

&emsp;&emsp;目标检测的任务是给定一张图片，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。通常用包围框来表示目标的位置，该框需紧凑地包围住目标的全部范围。

* 实例分割

&emsp;&emsp;实例分割的任务是在目标检测的基础上，找出各包围框内属于目标的像素点。

我们提供了分别用于完成目标检测和实例分割的神经网络模型YOLO V3以及Mask R-CNN，这两个模型的原理和可视化结果请参见[模型简介](/docs/2_模型简介/2_模型简介.md)，模型在COCO验证集上的精度表现如下：

* YOLO V3

* Mask R-CNN

| 主干网络             | 检测精度(Box AP) | 分割精度(Mask AP) |                           下载                           |
| :------------------ | :-------------: | :--------------: | :----------------------------------------------------------: |
| ResNet50-vd-FPN     |       39.8      |       35.4       | [模型参数](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar)|
| SENet154-vd-FPN     |       44.0      |       38.7       | [模型参数](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |
