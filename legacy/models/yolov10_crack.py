import torch
import torch.nn as nn
from .attention import CBAM


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class YOLOv10Crack(nn.Module):
    """
    YOLOv10裂缝检测模型（简化版）
    用于农田裂缝检测的三分类任务
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = nn.Sequential(
            ConvBNReLU(3, 32, 7, 2, 3),      # 320x320
            nn.MaxPool2d(2, 2),                # 160x160
            ConvBNReLU(32, 64, 3, 2, 1),       # 80x80
            ResidualBlock(64),
            ConvBNReLU(64, 128, 3, 2, 1),      # 40x40
            ResidualBlock(128),
            ConvBNReLU(128, 256, 3, 2, 1),     # 20x20
            CBAM(256),
            ResidualBlock(256)
        )
        
        # Neck (特征金字塔)
        self.neck = nn.Sequential(
            ConvBNReLU(256, 512, 3, 2, 1),     # 10x10
            CBAM(512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 20x20
            ConvBNReLU(512 + 256, 256, 3, 1, 1)
        )
        
        # Detection Head
        self.detection_head = nn.Sequential(
            ConvBNReLU(256, 128, 3, 1, 1),
            nn.Conv2d(128, 5 + num_classes, 1)  # [x, y, w, h, conf, class_scores]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        neck_features = self.neck(features)
        output = self.detection_head(neck_features)
        
        batch_size = output.size(0)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, -1, 5 + self.num_classes)
        
        return output
