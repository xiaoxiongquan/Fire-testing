# 实地操作
<font face="楷体" size=3>1. 数据处理
2. 模型训练
3. 模型导出


```python
# 首先安装两个依赖包
!pip install paddlex
!pip install paddle2onnx
```


```python
#解压数据集并将其移动至dataset中
!tar -xf /home/aistudio/data/data90352/fire_detection.tar 
```


```python
!mv VOC2020 dataset 
```


<font face="楷体" size=3>&emsp;&emsp;在本数据集中，由于文件名及文件内容不符合PaddleX所提供的数据集读取API，故需要对其进行处理。观察数据集可知，有两个问题：一为标注文件的文件名中存在空格，这极大地影响了PaddleX数据集读取；二为标注文件中的内容需要进行对应性修改。


```python
# 修改.xml文件名，去掉文件名中的空格
# -*- coding: utf-8 -*-
import os
#设定文件路径
jpg_path='dataset/JPEGImages/'
anno_path = 'dataset/Annotations/'
i=1
#对目录下的文件进行遍历
for file in os.listdir(jpg_path):
#判断是否是文件
    if os.path.isfile(os.path.join(jpg_path,file))==True:
#设置新文件名
        main = file.split('.')[0]
        if " " in main:
            new_main = main.replace(' ','')
            new_main_jpg = new_main + '.jpg'
            new_main_anno = new_main + '.xml'
            print(os.path.join(jpg_path,new_main_jpg))
            print(os.path.join(anno_path,new_main_anno))

#         new_name=file.replace(file,"rgb_%d.jpg"%i)
# #重命名
            os.rename(os.path.join(jpg_path,main+'.jpg'),os.path.join(jpg_path,new_main_jpg))
            os.rename(os.path.join(anno_path,main+'.xml'),os.path.join(anno_path,new_main_anno))
            i+=1
#结束
print ("End")
```


```python
# 这里修改.xml文件中的<path>元素
!mkdir dataset/Annotations1
import xml.dom.minidom
import os

path = r'dataset/Annotations'  # xml文件存放路径
sv_path = r'dataset/Annotations1'  # 修改后的xml文件存放路径
files = os.listdir(path)
cnt = 1

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
    for i in item:
        i.firstChild.data = '/home/aistudio/dataset/JPEGImages/' + str(cnt).zfill(6) + '.jpg'  # xml文件对应的图片路径

    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
    cnt += 1
```


```python
# 这里修改.xml文件中的<failname>元素
!mkdir dataset/Annotations2
import xml.dom.minidom
import os

path = r'dataset/Annotations1'  # xml文件存放路径
sv_path = r'dataset/Annotations2'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    names = root.getElementsByTagName('filename')
    a, b = os.path.splitext(xmlFile)  # 分离出文件名a
    for n in names:
        n.firstChild.data = a + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
```

<font face="楷体" size=3>下面删除在数据集处理过程中所生成的冗余文件，并将其更改为适合PaddleX的数据集格式。


```python
!rm -rf dataset/Annotations
!rm -rf dataset/Annotations1
!mv dataset/Annotations2 dataset/Annotations
```

<font face="楷体" size=3>&emsp;&emsp;PaddleX非常**贴心**地为开发者准备了数据集划分工具，免去了开发者多写几行代码的需求。这里我们设置训练集、验证集、测试集划分比例为7：2：1。


```python
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/ --val_value 0.2 --test_value 0.1
```


```python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), 
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(), 
    transforms.Resize(target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), 
        transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/train_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    transforms=train_transforms,
    parallel_method='thread',
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/val_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    parallel_method='thread',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.PPYOLO(num_classes=num_classes)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
```


```python
model.train(
    num_epochs=80,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    learning_rate=0.00125,
    save_interval_epochs=1,
    lr_decay_epochs=[40,60,70],
    save_dir='output/ppyolo',
    pretrain_weights=None,
    use_vdl=True)
```

## 模型预测


```python
import numpy as np
import cv2
import paddlex as pdx
import plistlib as plt
# 读取单张图片
image ='dataset/JPEGImages/01e999e4-17cf-43ae-bccc-3acb8d23c251.jpg'

model = pdx.load_model('output/ppyolo/best_model')

# 单张图片预测
result = model.predict(image)
# print("Predict Result: ", result,end=" ")
pdx.det.visualize(image, result, threshold=0.08, save_dir='./')

```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:64
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:40
    The behavior of expression A / B has been unified with elementwise_div(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_div(X, Y, axis=0) instead of A / B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/io.py:2302: UserWarning: This list is not set, Because of Paramerter not found in program. There are: create_parameter_0.w_0 create_parameter_1.w_0 create_parameter_2.w_0 create_parameter_3.w_0 create_parameter_4.w_0 create_parameter_5.w_0 create_parameter_6.w_0 create_parameter_7.w_0 create_parameter_8.w_0 create_parameter_9.w_0 create_parameter_10.w_0 create_parameter_11.w_0 create_parameter_12.w_0 create_parameter_13.w_0 create_parameter_14.w_0 create_parameter_15.w_0 create_parameter_16.w_0 create_parameter_17.w_0 create_parameter_18.w_0 create_parameter_19.w_0 create_parameter_20.w_0 create_parameter_21.w_0 create_parameter_22.w_0 create_parameter_23.w_0 create_parameter_24.w_0 create_parameter_25.w_0 create_parameter_26.w_0 create_parameter_27.w_0 create_parameter_28.w_0 create_parameter_29.w_0 create_parameter_30.w_0 create_parameter_31.w_0 create_parameter_32.w_0 create_parameter_33.w_0 create_parameter_34.w_0 create_parameter_35.w_0 create_parameter_36.w_0 create_parameter_37.w_0 create_parameter_38.w_0 create_parameter_39.w_0 create_parameter_40.w_0 create_parameter_41.w_0 create_parameter_42.w_0 create_parameter_43.w_0 create_parameter_44.w_0 create_parameter_45.w_0 create_parameter_46.w_0 create_parameter_47.w_0
      format(" ".join(unused_para_list)))


    2021-08-11 19:40:20 [INFO]	Model[PPYOLO] loaded.
    2021-08-11 19:40:20 [INFO]	The visualized result is saved as ./visualize_01e999e4-17cf-43ae-bccc-3acb8d23c251.jpg

