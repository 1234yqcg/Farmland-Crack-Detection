# 训练单张图片
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from utils.roboflow_dataset import RoboflowFarmlandDataset

def train_single_image():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLOv10Crack(
        num_classes=1,
        depth_multiple=0.33,
        width_multiple=0.25,
        use_attention=True
    ).to(device)

    pretrained_path = 'weights/yolov10n.pt'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")

    optimizer = AdamW(model.parameters(), lr=0.001)
    
    dataset_yaml_path = r'd:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection\data\dataset.yaml'
    
    dataset = RoboflowFarmlandDataset(
        data_yaml_path=dataset_yaml_path,
        split='train',
        image_size=(512, 512),
        augment=False,
        cache_images=False
    )

    print(f"Total training samples: {len(dataset)}")

    data = dataset[0]
    image = data['image'].unsqueeze(0).to(device)
    labels = data['labels']
    
    print(f"Image shape: {image.shape}")
    print(f"Labels: {labels}")
    
    feature_map_sizes = [(64, 64), (32, 32), (16, 16)]
    strides = [8, 16, 32]
    anchor_sizes = [
        (10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198),
        (373, 326), (10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90)
    ]
    
    all_anchor_centers = []
    anchor_idx = 0
    for level_idx, ((fh, fw), stride) in enumerate(zip(feature_map_sizes, strides)):
        for grid_y in range(fh):
            for grid_x in range(fw):
                cx = (grid_x + 0.5) * stride
                cy = (grid_y + 0.5) * stride
                all_anchor_centers.append((cx, cy, stride, anchor_idx))
                anchor_idx += 1
    
    target = labels[0]
    cls_id = int(target[0].item())
    x1, y1, x2, y2 = target[1].item(), target[2].item(), target[3].item(), target[4].item()
    gt_cx = (x1 + x2) / 2
    gt_cy = (y1 + y2) / 2
    
    print(f"\nTarget: class={cls_id}, box=[{x1}, {y1}, {x2}, {y2}], cx={gt_cx}, cy={gt_cy}")
    
    best_anchor = None
    best_dist = float('inf')
    for cx, cy, stride, idx in all_anchor_centers:
        dist = ((gt_cx - cx) ** 2 + (gt_cy - cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_anchor = (idx, cx, cy, stride)
    
    pos_indices = [best_anchor]
    
    print(f"Best anchor idx={best_anchor[0]}, dist={best_dist:.2f}")
    
    model.train()
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        outputs = model(image)
        
        b, total_anchors, c = outputs.shape
        num_classes = 1
        
        cls_channels = num_classes + 1
        reg_channels = c - cls_channels
        
        reg_max = reg_channels // 4
        target_reg_channels = 4 * reg_max
        
        cls_pred = outputs[:, :, :cls_channels]
        reg_pred = outputs[:, :, cls_channels:cls_channels + target_reg_channels]
        
        if epoch % 10 == 0:
            print(f"\n=== Epoch {epoch+1}/50 ===")
        
        pos_mask = torch.zeros(b * total_anchors, dtype=torch.bool, device=device)
        target_cls_map = torch.zeros(b * total_anchors, dtype=torch.long, device=device)
        target_box_map = torch.zeros(b * total_anchors, 4, device=device)
        
        idx, cx, cy, stride = pos_indices[0]
        if idx < total_anchors:
            aw, ah = anchor_sizes[0]
            l = (x1 - cx) * reg_max / aw
            t = (y1 - cy) * reg_max / ah
            r = (x2 - cx) * reg_max / aw
            b = (y2 - cy) * reg_max / ah
            
            pos_mask[idx] = True
            target_cls_map[idx] = cls_id
            target_box_map[idx] = torch.tensor([l, t, r, b], device=device)
        
        num_pos = pos_mask.sum()
        
        if num_pos > 0:
            cls_criterion = nn.CrossEntropyLoss(reduction='mean')
            cls_pred_pos = cls_pred[0, pos_mask, :num_classes]
            target_cls_pos = target_cls_map[pos_mask]
            cls_loss = cls_criterion(cls_pred_pos, target_cls_pos)
            
            obj_criterion = nn.BCEWithLogitsLoss(reduction='mean')
            obj_pred_pos = cls_pred[0, pos_mask, num_classes]
            obj_pred_neg = cls_pred[0, ~pos_mask, num_classes]
            obj_loss = (obj_criterion(obj_pred_pos, torch.ones_like(obj_pred_pos)) + 
                       obj_criterion(obj_pred_neg, torch.zeros_like(obj_pred_neg))) / 2
            
            reg_criterion = nn.SmoothL1Loss(reduction='mean')
            reg_pred_pos = reg_pred[0, pos_mask]
            reg_pred_pos = reg_pred_pos.view(-1, 4, reg_max)
            exp_scores = torch.exp(reg_pred_pos - reg_pred_pos.max(dim=2, keepdim=True)[0])
            reg_pred_decoded = (exp_scores * torch.arange(reg_max, device=device, dtype=torch.float32).view(1, 1, reg_max)).sum(dim=2)
            target_box_pos = target_box_map[pos_mask]
            reg_loss = reg_criterion(reg_pred_decoded, target_box_pos)
            
            loss = cls_loss + obj_loss + 5.0 * reg_loss
            
            if epoch % 10 == 0:
                print(f"Loss: cls={cls_loss:.4f}, obj={obj_loss:.4f}, reg={reg_loss:.4f}, total={loss.item():.4f}")
        else:
            loss = cls_pred.mean() * 0.01
            if epoch % 10 == 0:
                print(f"No positive! Random loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
    
    os.makedirs('outputs/exp_anchor_loss/weights', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/exp_anchor_loss/weights/test_model.pt')
    print("\nModel saved!")

if __name__ == '__main__':
    train_single_image()
