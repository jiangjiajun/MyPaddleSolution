# PaddleSolution

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/jiangjiajun/PaddleSolution) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle Enterprise Solution

PaddlePaddle企业解决方案

## 目录
* [1 任务描述](#1-任务描述)
* [2 模型简介](#2-模型简介)
  * [2.1 目标检测](#21-目标检测)
  * [2.2 实例分割](#22-实例分割)
* [3 模型训练](#3-模型训练)
  * [3.1 安装](#31-安装)
  * [3.2 数据准备](#32-数据准备)
  * [3.3 训练](#33-训练)
  * [3.4 评估](#34-评估)
  * [3.5 预测](#35-预测)
* [4 模型调优](#4-模型调优)

## 1 任务描述

&emsp;&emsp;目标检测的任务是给定一张图片，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。通常用包围框来表示目标的位置，该框需紧凑地包围住目标的全部范围。用户可以使用YOLO V3来完成此项任务。
  
  <div align="center">
    <img src="demo/000000509403_bbox.jpg" />
  </div>
  
  &emsp;&emsp;实例分割的任务是在目标检测的基础上，找出各包围框内属于目标的像素点。用户可以使用Mask R-CNN来完成此项任务。

  <div align="center">
    <img src="demo/000000509403_mask.jpg" />
  </div>

  &emsp;&emsp;接下来将简要介绍YOLOv3和Mask RCNN的原理。

## 2 模型简介

### 2.1 目标检测

&emsp;&emsp;基于卷积神经网络的目标检测方法大致可分为两类：二阶段检测器和一阶段检测器。

&emsp;&emsp;二阶段检测器第一步是提取出目标可能存在的区域，这些区域称为候选框，第二步是对各候选框进行种类识别和位置回归操作，分别得到候选框属于各个类别的得分以及针对每一种类别微调后的位置。经典模型有Faster R-CNN和Cascade R-CNN等。

&emsp;&emsp;一阶段检测器去掉了预先提取候选框这一步骤，让网络直接输出各目标的位置和类别信息。经典模型有SSD和YOLO V3等。

&emsp;&emsp;二阶段检测器的精度都比较高，但是对每个候选框都要进行分类和回归的这一步骤导致此类方法非常耗时。而一阶段检测器具有的优点是检测速度快，但精度较二阶段检测器的低。

&emsp;&emsp;YOLO V3是现阶段检测精度和效率都较高的模型，推荐用户使用YOLO V3来完成目标检测。YOLO V3的检测流程为：

### 2.2 实例分割

&emsp;&emsp;Mask R-CNN是经典的实例分割模型，基本思想是首先利用候选框生成网络得到目标在图像中的候选框，其次用候选框池化操作提取出特征层上属于各候选框的特征，最后将候选框特征输入给分类子网络、回归子网络和分割子网络得到像素级别的检测结果。如下图所示，Mask R-CNN主要包含四个部分：

&emsp;&emsp;1. 主干网络。主干网络生成输入图像的特征层，用于后续的候选框生成，以及候选框的位置回归、种类识别和像素点分割。

&emsp;&emsp;2. 候选框生成网络。以主干网络输出的特征层作为输入，输出图像上可能存在物体的区域，这些区域称为候选框。

&emsp;&emsp;3. 候选框池化操作。以候选框和主干网络输出的特征层作为输入，输出特征层中属于各候选框的特征。

&emsp;&emsp;4. 分类子网络、回归子网络和分割子网络。以候选框的特征作为输入，输出候选框属于各个类别的概率、针对每一种类别微调后的位置以及属于各个类别的像素点分割结果。


 <div align="center">
    <img src="demo/Mask R-CNN architecture.png" />
 </div>

## 3 模型训练

### 3.1 安装

&emsp;&emsp;运行PaddleSolution需预先安装PaddlePaddle和其他依赖项。

&emsp;&emsp;**(1) 环境要求**
  ```
  - Python2 或 Python3
  - CUDA >= 8.0
  - cuDNN >= 5.0
  - nccl >= 2.1.2
  ```

&emsp;&emsp;**(2) 安装PaddlePaddle**

&emsp;&emsp;请参照[PaddlePaddle安装指南](http://www.paddlepaddle.org.cn/)安装PaddlePaddle Fluid v.1.5或1.5之后的版本。请务必运行以下代码以确保PaddlePaddle安装成功以及所安装的版本不低于1.5。
  
  ```
  # 查看PaddlePaddle是否安装成功
  >>> import paddle.fluid as fluid 
  >>> fluid.install_check.run_check()

  # 查看所安装的版本
  python -c "import paddle; print(paddle.__version__)"
  ```

&emsp;&emsp;**(3) 安装其他依赖项**
  
  &emsp;&emsp;PaddleSolution的运行需要依赖[COCO-API](https://github.com/cocodataset/cocoapi)，请运行以下代码进行安装：
  
  ```
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  # 如果未安装cython，请运行下一行代码进行安装；否则，跳过
  pip install Cython
  # 如果要把COCO-API安装至global site-packages，请运行下一行代码进行安装；否则，跳过
  make install
  # 如果没有sudo权限或者不想安装至global site-packages，请运行下一行代码进行安装
  python setup.py install --user
  ```

&emsp;&emsp;**(4) 安装PaddleSolution**
  
  &emsp;&emsp;克隆模型库并进入到模型库的目录：
  ```
  cd <your_path/to/clone/PaddleSolution>
  git clone https://github.com/PaddlePaddle/PaddleSolution
  cd PaddleSolution
  ```
  
  &emsp;&emsp;安装Python依赖项：
  ```
  pip install -r requirements.txt
  ```
  
  &emsp;&emsp;确认测试样例可以正常运行：
  
  ```
  export PYTHONPATH=`pwd`:$PYTHONPATH
  python ppdet/modeling/tests/test_architectures.py
  ```
  
### 3.2 数据准备

&emsp;&emsp;PaddleSolution目前支持[COCO](http://cocodataset.org)数据集和自定义数据集。请按照以下指南准备所需的数据集。

**3.2.1 COCO数据集**

  &emsp;&emsp;如果已经下载过COCO数据集，请按照以下步骤将存放COCO数据集的目录链接到当前PaddleSolution的目录下:

  ```
  # <path/to/coco>和<path/to/PaddleSolution>都应该是绝对路径
  ln -sf <path/to/coco> <path/to/PaddleSolution>/dataset/coco
  ```
  
  &emsp;&emsp;如果想自己下载COCO数据集，可运行以下代码：

  ```
  ./dataset/coco/download.sh
  ```

  &emsp;&emsp;如果未下载COCO数据集或目录`dataset/coco`下没有数据集，PaddleSolution会自动下载[COCO-2017](http://images.cocodataset.org)数据集并解压至`~/.cache/paddle/dataset/`，模型训练时会使用该目录下的数据集。
  
**3.2.2 自定义数据集**

  &emsp;&emsp;目前PaddleSolution支持用户使用自定义数据集。用户在采集完图片之后，先使用数据标注工具LabelMe完成数据标注，再使用PaddleSolution提供的[数据转换脚本]()将LabelMe产出的数据格式转换为模型训练时所需的数据格式。
  
  &emsp;&emsp;**(1) LabelMe安装方法**
  
  &emsp;&emsp;[LabelMe](https://github.com/wkentaro/labelme)支持在Windows/MacOS/Ubuntu三个系统上使用，且三个系统下的标注格式是一样。

  >* Windows
    * 参考[安装文档](https://docs.anaconda.com/anaconda/install/windows/)安装Anaconda, 安装后打开Anaconda Navigator创建一个新的环境并进入该环境。

    * 安装pyqt:
    ```
    # 如果使用python2
    pip install pyqt
    # 如果使用python3
    pip install pyqt5
    ```
    * 安装labelme:
    ```
    pip install labelme
    ```
    * 安装pillow:
    ```
    python -m pip install pillow
    ```
    * 运行:
    ```
    labelme
      ```
  * MacOS
    
    使用[安装指南](https://github.com/wkentaro/labelme)完成MacOS下LableMe的安装。
    
  * Ubuntu
    
    使用[安装指南](https://github.com/wkentaro/labelme)完成Ubuntu下LableMe的安装。
    
  &emsp;&emsp;**(2) 将LabelMe产出的数据格式转换为PaddleSolution所需的格式**
  
  &emsp;&emsp;运行以下代码，可以自动将数据划分为训练集、验证集和测试集，同时文件目录组织结构与COCO标准数据集格式一致。终端同时会输出目标类别数量，请将该数值记下，后续模型训练会使用该数值。
   
  ```
  # train_proportion、val_proportion和test_proportion为可选参数，分别代表训练集、验证集和测试集占总共数据的比例，三者之和必须等于1
  # json_input_dir：LabelMe标注工具产出的json文件的存储路径
  # image_input_dir：json_input_dir中各json文件所对应图像的存储路径，该路径中各文件的命名与json_input_dir中的必须一一对应，且数量相同
  # output_dir：转换后的数据集的存储路径
  python labelme2coco.py --json_input_dir=<path/to/json_files>/json_files/ 
                      --image_input_dir=<path/to/img_files>/img_files/ 
                      --output_dir=<path/to/coco>
                      --train_proportion=0.7
                      --val_proportion=0.2 
                      --test_proportion=0.1
  ```
  
### 3.3 训练

### 3.4 评估

### 3.5 预测

## 4 模型调优
