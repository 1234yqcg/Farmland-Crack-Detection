import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class DecoupledHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)
        
        self.dfl = DFL(reg_max)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_feat = self.cls_conv(x)
        cls_pred = self.cls_pred(cls_feat)
        
        reg_feat = self.reg_conv(x)
        reg_pred = self.reg_pred(reg_feat)
        
        return cls_pred, reg_pred
