# PaddleSolution

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/jiangjiajun/PaddleSolution) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle提供了针对视觉任务的端到端企业解决方案，覆盖了从数据准备到模型部署的整个流程。在此Repo中，针对每种视觉任务，我们提供了现阶段精度和效率皆优的神经网络模型，并展示了如何准备数据，如何使用PaddlePaddle完成模型的训练、压缩和部署。


## 目录
* [1 简介](#1-简介)
* [2 数据准备](#2-数据准备)
* [3 模型训练](#3-模型训练)
  * [3.1 安装](#31-安装)
  * [3.2 训练](#32-训练)
  * [3.3 评估](#33-评估)
  * [3.4 预测](#34-预测)
  * [3.5 调优](#35-调优)
* [4 模型压缩](#4-模型压缩)
* [5 模型部署](#5-模型部署)
## 1 简介

PaddleSolution的目标是通过提供针对视觉任务的端到端解决方案，帮助用户打通从准备数据，训练模型，到压缩和部署模型的全部流程。目前支持的视觉任务有：目标检测和实例分割。针对这两种任务，我们提供了神经网络模型YOLO V3以及Mask R-CNN，这两个模型的原理和可视化结果请参见[模型简介](./docs/1_简介/模型简介.md)，模型在COCO验证集上的精度表现如下：

* YOLO V3

* Mask R-CNN

| 主干网络             | 检测精度(Box AP) | 分割精度(Mask AP) |                           下载                           |
| :------------------ | :-------------: | :--------------: | :----------------------------------------------------------: |
| ResNet50-vd-FPN     |       39.8      |       35.4       | [模型参数](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar)|
| SENet154-vd-FPN     |       44.0      |       38.7       | [模型参数](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |

## 2 数据准备

PaddleSolution目前支持[COCO](http://cocodataset.org)数据集格式。若不使用COCO数据集，用户需预先采集好用于训练、评估和预测的图片，并使用数据标注工具[LabelMe]((https://github.com/wkentaro/labelme))完成数据标注，最后用我们提供的[数据转换脚本]()将LabelMe产出的数据格式转换为模型训练时所需的数据格式。具体流程请参见[数据准备.md]()。

## 3 模型训练

### 3.1 安装

运行PaddleSolution对环境有所要求，且需预先安装PaddlePaddle和其他依赖项。具体流程请参见[模型安装.md]()。

### 3.2 训练

#### 3.3.1 目标检测

#### 3.3.2 实例分割

选择不同的主干网络，Mask R-CNN的分割精度有所差别。推荐用户使用主干网络为ResNet50-vd-FPN的Mask R-CNN来完成实例分割，如果想要更高的精度，可以选择SENet154-vd-FPN作为主干网络，但运行速度会稍慢些。

主干网络为ResNet50-vd-FPN的配置文件为[mask_rcnn_r50_vd_fpn.yml]()，该配置文件的部分参数是针对使用8块显卡训练COCO数据集所设置的，运行前请根据实际情况调整这些参数，具体的调整方法请参见[Mask R-CNN参数调整.md]()。

调整好参数之后，请参见[Mask R-CNN训练.md]()进行训练。


### 3.3 评估

目前仅支持使用单块显卡进行评估，模型参数的路径通过[mask_rcnn_r50_vd_fpn.yml]()中`weights`来指定。
```
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/mask_rcnn_r50_vd_fpn.yml
```
运行结束后，终端会输出多项指标的数值，各项指标的具体含义请参考[COCO数据集官方文档](http://cocodataset.org/#detection-eval)。这里只考虑`Average Precision(AP) @ [ IoU=0.50:0.95 | area= all | maxDets=100 ]`，该项指标的数值越高，表示模型的精度越高。

### 3.4 预测

训练好的的模型参数可用于预测单张图片或者批量图片，具体的步骤以及可视化结果请参见[预测.md]()。

### 3.5 调优

[配置文件]()中各参数的默认值对于COCO数据集来说是最优的，这些设定值对于其他数据集来说可能不是最优的。用户在训练自定义数据集时，可以调整这些参数，以期获得精度或效率的提升。具体的调整策略请参见[模型调优.md]()。

## 4 模型压缩

## 5 模型部署
