import torch
import torch.nn as nn
from typing import List

from ..backbone.csp_darknet import ConvBlock, C2f

class PANet(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(in_channels[2] + in_channels[1], out_channels, 1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p3 = C2f(in_channels[1] + in_channels[0], out_channels, 1)
        
        self.down1 = ConvBlock(out_channels, out_channels, 3, 2, 1)
        self.c2f_n3 = C2f(out_channels * 2, out_channels, 1)
        
        self.down2 = ConvBlock(out_channels, out_channels, 3, 2, 1)
        self.c2f_n4 = C2f(out_channels * 2, out_channels, 1)
        
        self.out_channels = [out_channels, out_channels, out_channels]
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features
        
        x = self.up1(p5)
        x = torch.cat([x, p4], dim=1)
        n4 = self.c2f_p4(x)
        
        x = self.up2(n4)
        x = torch.cat([x, p3], dim=1)
        n3 = self.c2f_p3(x)
        
        x = self.down1(n3)
        x = torch.cat([x, n4], dim=1)
        n4_out = self.c2f_n3(x)
        
        x = self.down2(n4_out)
        x = torch.cat([x, p5], dim=1)
        n5_out = self.c2f_n4(x)
        
        return [n3, n4_out, n5_out]
