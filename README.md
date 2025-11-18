## 1.下载数据集

下载BUSI等数据集，并按照如下示例存放：

    data
    └── BUSI
        ├── images
        └── masks

## 2. 数据集预处理

利用"./process/trans_BUSI.py"对BUSI数据集进行预处理，得到train.txt、val.txt和test.txt：

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

## Cite

    @ARTICLE{11104828,
      author={Xiong, Youqiang and Shu, Xiu and Liu, Qiao and Yuan, Di},
      journal={IEEE Transactions on Consumer Electronics}, 
      title={HCMNet: A Hybrid CNN-Mamba Network for Breast Ultrasound Segmentation for Consumer Assisted Diagnosis}, 
      year={2025},
      volume={71},
      number={3},
      pages={8045-8054},
      keywords={Feature extraction;Image segmentation;Transformers;Computational modeling;Computer architecture;Adaptation models;Wavelet transforms;Noise;Decoding;Accuracy;Breast ultrasound images segmentation;hybrid network;wavelet feature;adaptive feature fusion},
      doi={10.1109/TCE.2025.3593784}}
