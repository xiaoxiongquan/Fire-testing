# PaddleX系列原来星星之火也可以燎原



# 一、项目背景

因为现在火灾问题备受关注，火灾给人们带来的损失是巨大的，所以我想做这么一个火灾检测的模型然后可以以后拿来检测火灾。

可附上效果展示。

# 二、数据集简介

介绍你的项目使用了什么数据集，一共有多少条数据，数据是什么样的等等。此处可细分，如下所示：

## 1.数据加载和预处理

```python
# 数据的加载
!tar -xf /home/aistudio/data/data90352/fire_detection.tar 
```

#预处理
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


## 数据集分列
```python
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/ --val_value 0.2 --test_value 0.1
```

Train samples: 1443


Eval samples: 411


Test samples: 205



## 训练数据集
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

```


# 三、模型选择和开发



## 1.模型训练


```python
model.train(
    num_epochs=80,
    train_dataset=train_dataset,   # 训练数据集
    train_batch_size=16,        # 训练使用的批大小
    eval_dataset=eval_dataset,    # 评估数据集
    learning_rate=0.00125,
    save_interval_epochs=1,
    lr_decay_epochs=[40,60, 70],
    save_dir='output/ppyolo',
    pretrain_weights=None,
    use_vdl=True)
 
 ```

## 2.模型预测


```python
# 读取单张图片
image ='dataset/JPEGImages/01f5a694-f3b6-4032-b89d-70e29a65363b.jpg'

# 单张图片预测
result = model.predict(image)

# 可视化结果
pdx.det.visualize(image, result, threshold=0.08, save_dir='./')
```

# 四、效果展示

可以检验视频以及图片中的ROI,然后标注出来，可能并没有那么精确。
![](https://ai-studio-static-online.cdn.bcebos.com/2afc4e08bae2414196168bba5cbdc852e0b3259c37d3461a9837577d57f93ea8)



# 五、总结与升华

刚开始标注文件也不懂，后来就看看官网文档，参考别人的优质项目，借鉴他们的分类文件方式，然后就是在训练集那块也是看了官网文档知道怎么调整训练集，采用官方文档说的一些图片增强的方式进行配置，最后训练模型的时候，batchsize和训练轮数和学习力已经下降学习力那一块弄了好久，最后也还是没有弄出最优质的方式。

最后我想说感谢 Ai Studio 给我提供了这么好的一个机会，可以向那么多大佬学习，我一定会好好在这个平台汲取知识的。

最后一句话总结你的项目 第一次使用可以说相当的不理想，但是我相信有这么一个好的平台，里面还有汇聚了这么多大佬，往后的日子都可以努力向他们学习，慢慢变强，早日成为社区有价值的一份子。

# 个人简介

小菜鸡
