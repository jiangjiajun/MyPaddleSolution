# PaddleSolution

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/jiangjiajun/PaddleSolution) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle Enterprise Solution

PaddlePaddle企业解决方案

## 目录
* [任务描述](#任务描述)
* [模型简介](#模型简介)
  * [目标检测](#目标检测)
  * [实例分割](#实例分割)
* [模型训练](#模型训练)
  * [安装](#安装)
  * [数据准备](#数据准备)
  * [训练](#训练)
  * [评估](#评估)
  * [预测](#评估)
* [模型调优](#模型调优)

## 任务描述

  目标检测的任务是给定一张图片，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。通常用包围框来表示目标的位置，该框需紧凑地包围住目标的全部范围。用户可以使用YOLO V3来完成此项任务。
  
  实例分割的任务是在目标检测的基础上，找出各包围框内属于目标的像素点。用户可以使用Mask R-CNN来完成此项任务。

<div align="center">
  <img src="demo/000000570688.jpg" />
</div>

  接下来将简要介绍YOLOv3和Mask RCNN的原理。

## 模型简介

### 目标检测

基于卷积神经网络的目标检测方法大致可分为两类：二阶段检测器和一阶段检测器。

二阶段检测器第一步是提取出目标可能存在的区域，这些区域称为候选框，第二步是对各候选框进行种类识别和位置回归操作，分别得到候选框属于各个类别的得分以及针对每一种类别微调后的位置。经典模型有Faster R-CNN和Cascade R-CNN等。

一阶段检测器去掉了预先提取候选框这一步骤，让网络直接输出各目标的位置和类别信息。经典模型有SSD和YOLO V3等。

二阶段检测器的精度都比较高，但是对每个候选框都要进行分类和回归的这一步骤导致此类方法非常耗时。而一阶段检测器具有的优点是检测速度快，但精度较二阶段检测器的低。

YOLO V3是现阶段检测精度和效率都较高的模型，推荐用户使用YOLO V3来完成目标检测。YOLO V3的检测流程为：

### 实例分割

## 模型训练

### 安装

### 数据准备

### 训练

### 评估

### 预测

## 模型调优
