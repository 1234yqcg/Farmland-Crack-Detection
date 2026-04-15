import torch
import torch.nn as nn
from typing import List

from ..backbone.csp_darknet_fixed import ConvBlock

class PANet(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        # in_channels = [p3, p4, p5] = [64, 128, 256] for width=0.50
        # 动态适配输入通道
        
        # 自顶向下路径
        # 1. p5上采样 + p4 -> out_channels
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_p4 = ConvBlock(in_channels[2] + in_channels[1], out_channels, 1, 1, 0)
        
        # 2. n4 + p3 -> out_channels
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_p3 = ConvBlock(out_channels + in_channels[0], out_channels, 1, 1, 0)
        
        # 自底向上路径
        # 3. n3下采样 + n4 -> out_channels
        self.down1 = ConvBlock(out_channels, out_channels, 3, 2, 1)
        self.conv_n3 = ConvBlock(out_channels + out_channels, out_channels, 1, 1, 0)
        
        # 4. n4_out下采样 + p5 -> out_channels
        self.down2 = ConvBlock(out_channels, out_channels, 3, 2, 1)
        self.conv_n4 = ConvBlock(out_channels + in_channels[2], out_channels, 1, 1, 0)
        
        self.out_channels = [out_channels, out_channels, out_channels]
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features
        
        # 自顶向下路径
        x = self.up1(p5)
        x = torch.cat([x, p4], dim=1)
        n4 = self.conv_p4(x)
        
        x = self.up2(n4)
        x = torch.cat([x, p3], dim=1)
        n3 = self.conv_p3(x)
        
        # 自底向上路径
        x = self.down1(n3)
        x = torch.cat([x, n4], dim=1)
        n4_out = self.conv_n3(x)
        
        x = self.down2(n4_out)
        x = torch.cat([x, p5], dim=1)
        n5_out = self.conv_n4(x)
        
        return [n3, n4_out, n5_out]