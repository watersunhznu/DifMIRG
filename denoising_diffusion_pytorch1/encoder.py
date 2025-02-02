#1113前版
# import torch
# from torch import nn
# import torchvision.models as models
# from einops import rearrange
#
# class Encoder(nn.Module):
#     def __init__(self, word_dim, from_x=-1):
#         super(Encoder, self).__init__()
#         self.from_x = from_x
#         resnet = models.resnet34(pretrained=False)
#         modules = list(resnet.children())[:from_x]
#         self.resnet = nn.Sequential(*modules)
#         self.linear = nn.Linear(512, word_dim)
#
#     def forward(self, x):
#         x = self.resnet(x).squeeze()  # (batch_size, enc_dim, enc_img_size, enc_img_size)
#         if self.from_x != -1:
#             x = rearrange(x, 'b d h w ->b (h w) d')
#         # x = x.permute(0, 2, 3, 1)
#         x = self.linear(x)
#         return x
#
# if __name__ == '__main__':
#     encoder = Encoder(256, -2)
#     img = torch.randn(16, 3, 256, 256)
#     out = encoder(img)
#     print(out.shape)
#












#1113改版

import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange


class FPN(nn.Module):
    def __init__(self, out_channels):
        super(FPN, self).__init__()
        # lateral3 的输入通道数应为 128，lateral4 的输入通道数改为 512
        self.lateral3 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, out_channels, kernel_size=1)

        self.top_down4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c3, c4):
        # 逐层处理 c4 和 c3
        p4 = self.lateral4(c4)  # 转换成 out_channels 通道
        p3 = self.lateral3(c3)  # 转换成 out_channels 通道

        # 上采样 p4 使其尺寸与 p3 匹配
        p4_upsampled = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = p3 + p4_upsampled

        # 平滑处理
        p3 = self.top_down4(p3)

        # 最终上采样 p3 以确保尺寸一致
        p3_upsampled = F.interpolate(p3, size=(32, 32), mode="nearest")
        p4_upsampled = F.interpolate(p4, size=(32, 32), mode="nearest")

        return p3_upsampled, p4_upsampled


class Encoder(nn.Module):
    def __init__(self, word_dim, out_channels=256):
        super(Encoder, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        #resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Extract feature layers for FPN: c3 and c4
        self.layer3 = nn.Sequential(*list(resnet.children())[:6])  # Output 256 channels
        self.layer4 = nn.Sequential(*list(resnet.children())[6:8])  # Output 512 channels

        self.fpn = FPN(out_channels=out_channels)
        self.linear = nn.Linear(14 * 32 * 32, word_dim)  # Adjusted for fused features

    def forward(self, x):
        c3 = self.layer3(x)  # 获取 layer3 输出
        c4 = self.layer4(c3)  # 获取 layer4 输出

        # 使用 FPN 进行多尺度特征融合
        p3, p4 = self.fpn(c3, c4)

        # 拼接处理后的特征
        fused_features = torch.cat([p3, p4], dim=1)

        # 展平并传入 linear 层
        x = self.linear(fused_features.flatten(start_dim=1))

        return x



if __name__ == '__main__':
    encoder = Encoder(64, -2)
    img = torch.randn(16, 3, 256, 256)
    out = encoder(img)
    print(out.shape)

