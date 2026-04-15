import torch
import torch.nn as nn
from typing import List, Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, groups: int = 1, 
                 activation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.cv2 = ConvBlock(hidden_channels * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))

class C2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = out_channels // 2
        self.cv1 = ConvBlock(in_channels, 2 * self.c, 1, 1, 0)
        self.cv2 = ConvBlock((2 + n) * self.c, out_channels, 1, 1, 0)
        self.m = nn.ModuleList(
            ConvBlock(self.c, self.c, 3, 1, 1) for _ in range(n)
        )
        self.shortcut = shortcut
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 修复通道分割问题
        y1 = self.cv1(x)
        # 确保通道数能被2整除
        channels = y1.size(1)
        if channels % 2 != 0:
            # 如果通道数不能被2整除，调整通道数
            y1 = y1[:, :channels-1, :, :]  # 去掉最后一个通道
            channels = channels - 1
        
        y = list(y1.chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CSPDarknet(nn.Module):
    def __init__(self, depth_multiple: float = 0.33, 
                 width_multiple: float = 0.50):
        super().__init__()
        
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [3, 6, 9, 3]
        
        channels = [int(c * width_multiple) for c in base_channels]
        depths = [int(d * depth_multiple) for d in base_depths]
        
        self.stem = ConvBlock(3, channels[0], 3, 2, 1)
        
        self.stage1 = nn.Sequential(
            ConvBlock(channels[0], channels[1], 3, 2, 1),
            C2f(channels[1], channels[1], depths[0])
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(channels[1], channels[2], 3, 2, 1),
            C2f(channels[2], channels[2], depths[1])
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(channels[2], channels[3], 3, 2, 1),
            C2f(channels[3], channels[3], depths[2])
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(channels[3], channels[4], 3, 2, 1),
            C2f(channels[4], channels[4], depths[3]),
            SPPF(channels[4], channels[4], 5)
        )
        
        self.out_channels = channels[1:4]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return [p2, p3, p4]
