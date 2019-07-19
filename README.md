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

#### 3.2.1 COCO数据集

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
  
#### 3.2.2 自定义数据集（#待补充检测和分割的标注区别）

  &emsp;&emsp;目前PaddleSolution支持用户使用自定义数据集。用户在采集完图片之后，先使用数据标注工具LabelMe完成数据标注，再使用PaddleSolution提供的[数据转换脚本]()将LabelMe产出的数据格式转换为模型训练时所需的数据格式。
  
  &emsp;&emsp;**(1) LabelMe安装方法**
  
  &emsp;&emsp;[LabelMe](https://github.com/wkentaro/labelme)支持在Windows/MacOS/Ubuntu三个系统上使用，且三个系统下的标注格式是一样。

  * Windows下的安装
  
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
  * MacOS下的安装
    
  使用[安装指南](https://github.com/wkentaro/labelme)完成MacOS下LableMe的安装。
    
  * Ubuntu下的安装
    
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
#### 3.3.1 目标检测

#### 3.3.2 实例分割
&emsp;&emsp;选择不同的主干网络，Mask R-CNN的分割精度有所差别。推荐用户使用主干网络为ResNet50-vd-FPN的Mask R-CNN来完成实例分割，如果想要更高的精度，可以选择SENet154-vd-FPN作为主干网络，但运行速度会稍慢些。

&emsp;&emsp;**(1) 参数调整**

&emsp;&emsp;主干网络为ResNet50-vd-FPN的配置文件为[mask_rcnn_r50_vd_fpn.yml]()，该配置文件的部分参数是针对使用8块显卡训练COCO数据集所设置的，运行前请根据实际情况**调整这些参数**：

* max_iters：总迭代次数
  * 8块显卡训练COCO数据集时总迭代次数为360000次，令显卡数量为`n(n<=8)`，最大迭代次数应设置为`360000*(8/n)`次。例如，4块显卡训练时设置为`720000`次，1块显卡训练时设置为`2880000`次。
* snapshot_iter：每迭代一定次数，就保存一次模型参数，默认设置为10000次，可调整。
* save_dir：保存模型的路径，默认为output，可自定义。
* weights：评估和测试阶段所需的模型参数的路径。默认为训练时最后一次迭代时保存的模型参数，可自定义。
* num_classes：数据集中目标类别的总数。默认设置为81类，如果使用自定义数据集，需要修改该值。
* LearningRate/base_lr：训练时的基础学习率
  * 8块显卡训练COCO数据集时基础学习率为0.01，令显卡数量为`n(n<=8)`，基础学习率应设置为`0.01/(8/n)`。例如，4块显卡训练时设置为`0.005`，1块显卡训练时设置为`0.00125`。
* LearningRate/milestones：当迭代次数达到一定值时，学习率衰减一次
  * 8块显卡训练COCO数据集时，当迭代次数到达240000次时，学习率由0.01衰减至0.001，当迭代次数到达320000次时，学习率由0.001衰减至0.0001。令显卡数量为`n(n<=8)`，第一次衰减时的迭代次数应设置为`240000*(8/n)`次，第二次衰减时的迭代次数应设置为`320000*(8/n)`。例如，4块显卡训练时设置为`[480000, 640000]`，1块显卡训练时设置为`[1920000, 2560000]`。
* MaskRCNNTrainFeed/dataset：训练集
  * dataset_dir: 训练集的存储路径。默认设置为dataset/coco，如果未将数据集存放至该路径下或未建立链接，可修改。
  * image_dir： 训练集中图片的存储路径。默认设置为train2017。
  * annotation： 训练集中真值的存储路径。默认设置为annotations/instances_train2017.json。
* MaskRCNNEvalFeed/dataset：验证集
  * dataset_dir: 验证集的存储路径。默认设置为dataset/coco，如果未将数据集存放至该路径下或未建立链接，可修改。
  * image_dir： 验证集中图片的存储路径。默认设置为val2017，如果想评估测试集的精度，可以修改为test2017。
  * annotation： 验证集中真值的存储路径。默认设置为annotations/instances_val2017.json，如果想评估测试集的精度，可以修改为annotations/instances_test2017.json。

&emsp;&emsp;**(2) 训练**

* 训练前需开启以下标志以确保有足够的显存：
```
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
```

* 同时指定显卡的序号

&emsp;&emsp;以使用序号为0的显卡为例，指定单块显卡训练方式（运行前请先用nvidia-smi命令查看哪块显卡是空闲的）：
```
export CUDA_VISIBLE_DEVICES=0
```
&emsp;&emsp;以使用8块显卡为例，指定多块显卡训练方式：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

* 开始训练：
```
python tools/train.py -c configs/mask_rcnn_r50_vd_fpn.yml
```
&emsp;&emsp;如果训练中断后，想从上次保存模型时的迭代次数开始训练，请加上`-r`来指定上次保存的模型的存储路径:
```
python tools/train.py -c configs/mask_rcnn_r50_vd_fpn.yml -r output/mask_rcnn_r50_vd_fpn/XXXX/
```
&emsp;&emsp;如果想在训练过程中同时评估每次保存下载的模型参数，请加上`--eval`：
```
python tools/train.py -c configs/mask_rcnn_r50_vd_fpn.yml --eval
```

### 3.4 评估

&emsp;&emsp;目前仅支持使用单块显卡进行评估，模型参数的路径通过[mask_rcnn_r50_vd_fpn.yml]()中`weights`来指定。
```
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/mask_rcnn_r50_vd_fpn.yml
```

### 3.5 预测

* 单张图片的预测，通过添加`--infer_img`指定该图片的路径：
```
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/mask_rcnn_r50_vd_fpn.yml --infer_img=demo/000000570688.jpg
```
* 批量图片的预测，通过添加`--infer_dir`指定存放批量图片的文件夹路径：
```
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/mask_rcnn_r50_vd_fpn.yml --infer_dir=demo
```
&emsp;&emsp;可视化结果默认存放在`output`下，通过添加`--save_file`自定义存放路径。
```
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/mask_rcnn_r50_vd_fpn.yml --infer_img=demo/000000570688.jpg --save_file=<path/to/save/file>
```
* 模型保存
&emsp;&emsp;通过添加`--save_inference_model`来保存预测模型，该模型在PaddlePaddle预测库中能够直接被导入而不需要再重新组网。
```
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/mask_rcnn_r50_vd_fpn.yml --infer_img=demo/000000570688.jpg \
                      --save_inference_model
```

## 4 模型调优

