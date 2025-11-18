## 1.下载BUSI等数据集，并按照如下示例存放：

    data
    └── BUSI
        ├── images
        └── masks

## 2.利用"./process/trans_BUSI.py"对BUSI数据集进行预处理，得到train.txt、val.txt和test.txt：

    data
    └── BUSI
        ├── images
        ├── masks
        ├── train.txt
        ├── val.txt
        └── test.txt

## 3.配置python环境
    
PyTorch使用2.1版本即可，其余的缺啥装啥。

## 4. Train
在train.py中修改模型及相关参数后直接执行train.py文件即可。
