import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积层 => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器 (下采样)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # 解码器 (上采样)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2) # 转置卷积上采样
            )
            self.ups.append(DoubleConv(feature*2, feature))  # 拼接后通道数翻倍

        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 下采样路径
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层处理
        x = self.bottleneck(x)

        # 反转跳跃连接顺序（从深层到浅层）
        skip_connections = skip_connections[::-1]

        # 上采样路径（步长为2处理转置卷积和双卷积）
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 转置卷积上采样
            
            # 获取对应的跳跃连接
            skip = skip_connections[idx//2]
            
            # 调整尺寸（处理奇数尺寸情况）
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # 拼接跳跃连接
            concat = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](concat)  # 双卷积处理

        out = self.final_conv(x)

        return F.sigmoid(out)


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    x = torch.randn((1, 3, 256, 256)).to("cuda:0")
    model = Unet().to("cuda:0")
    y = model(x)
    print(y.shape)
    MACs, Params = profile(model, inputs=(x,), verbose=False)
    Flops, Params = clever_format([MACs * 2, Params], '%.2f')
    print(f"Flops:{Flops}")
    print(f"Params:{Params}")
