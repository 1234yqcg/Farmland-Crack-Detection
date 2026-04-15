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
        # 计算中间通道数：输出通道的一半
        c = out_channels // 2
        
        # 确保中间通道数是正数且至少为 1
        c = max(1, c)
        
        # 计算cv1的输出通道数（应该是 2*c）
        cv1_out_channels = 2 * c
        
        # 计算cv2的输入通道数
        cv2_in_channels = (2 + n) * c
        
        self.cv1 = ConvBlock(in_channels, cv1_out_channels, 1, 1, 0)
        self.cv2 = ConvBlock(cv2_in_channels, out_channels, 1, 1, 0)
        self.m = nn.ModuleList(
            ConvBlock(c, c, 3, 1, 1) for _ in range(n)
        )
        self.shortcut = shortcut
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过 1x1 卷积
        y1 = self.cv1(x)
        
        # 分割成两部分
        y = list(y1.chunk(2, 1))
        
        # 添加额外的分支
        y.extend(m(y[-1]) for m in self.m)
        
        # 拼接所有分支
        return self.cv2(torch.cat(y, 1))

class CSPDarknet(nn.Module):
    def __init__(self, depth_multiple: float = 0.33, 
                 width_multiple: float = 0.50):
        super().__init__()
        
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [3, 6, 9, 3]
        
        channels = [int(c * width_multiple) for c in base_channels]
        depths = [int(d * depth_multiple) for d in base_depths]
        
        # 确保所有通道数都是偶数
        channels = [(c + 1) if c % 2 != 0 else c for c in channels]
        
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
        
        self.out_channels = channels[2:5]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return [p2, p3, p4]