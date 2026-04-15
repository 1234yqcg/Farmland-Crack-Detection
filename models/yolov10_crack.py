import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from .backbone import CSPDarknet
from .neck import PANet
from .head import DecoupledHead
from .attention import CBAM

class YOLOv10Crack(nn.Module):
    def __init__(self, num_classes: int = 3, 
                 depth_multiple: float = 0.33,
                 width_multiple: float = 0.50,
                 use_attention: bool = True,
                 reg_max: int = 16):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.reg_max = reg_max
        
        self.backbone = CSPDarknet(depth_multiple, width_multiple)
        
        backbone_channels = self.backbone.out_channels
        
        self.neck = PANet(backbone_channels, 256)
        
        if use_attention:
            self.attention = nn.ModuleList([
                CBAM(256) for _ in range(3)
            ])
        
        self.heads = nn.ModuleList([
            DecoupledHead(256, num_classes, reg_max=reg_max) for _ in range(3)
        ])
        
        self.strides = [8, 16, 32]
    
    def forward(self, x: torch.Tensor, return_raw: bool = False) -> Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        features = self.backbone(x)
        neck_features = self.neck(features)
        
        outputs = []
        for i, (feat, head) in enumerate(zip(neck_features, self.heads)):
            if self.use_attention:
                feat = self.attention[i](feat)
            cls_pred, reg_pred = head(feat)
            outputs.append((cls_pred, reg_pred))
        
        if self.training or return_raw:
            return outputs
        
        return self._decode_outputs(outputs, x.shape)
    
    def _decode_outputs(self, outputs: List, input_shape: tuple) -> torch.Tensor:
        batch_size = input_shape[0]
        all_predictions = []
        
        for i, (cls_pred, reg_pred) in enumerate(outputs):
            b, c, h, w = cls_pred.shape
            stride = self.strides[i]
            
            # Process class predictions
            cls_pred = cls_pred.view(b, self.num_classes, -1).permute(0, 2, 1).sigmoid()
            
            # Process regression predictions (DFL)
            # reg_pred: [b, 4*reg_max, h, w]
            reg_pred = reg_pred.view(b, 4, self.reg_max, h, w).permute(0, 3, 4, 1, 2)
            # Softmax along reg_max dimension
            reg_pred = reg_pred.softmax(dim=-1)
            # Calculate expectation: sum(prob * value)
            # [b, h, w, 4, reg_max] * [reg_max] -> [b, h, w, 4]
            dfl_values = torch.arange(self.reg_max, device=reg_pred.device, dtype=torch.float32)
            reg_pred = (reg_pred * dfl_values).sum(dim=-1)
            
            # Reshape to [b, h*w, 4]
            reg_pred = reg_pred.view(b, -1, 4)
            
            # Create grid
            yv, xv = torch.meshgrid(torch.arange(h, device=reg_pred.device), torch.arange(w, device=reg_pred.device), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2).float() + 0.5
            
            # Decode boxes to xyxy
            # reg_pred is [l, t, r, b] relative to grid cell
            x1 = (grid[..., 0] - reg_pred[..., 0]) * stride
            y1 = (grid[..., 1] - reg_pred[..., 1]) * stride
            x2 = (grid[..., 0] + reg_pred[..., 2]) * stride
            y2 = (grid[..., 1] + reg_pred[..., 3]) * stride
            
            boxes = torch.stack([x1, y1, x2, y2], -1)
            
            all_predictions.append(torch.cat([boxes, cls_pred], dim=-1))
        
        return torch.cat(all_predictions, dim=1)
