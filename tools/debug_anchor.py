import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from utils.roboflow_dataset import RoboflowFarmlandDataset

def train_single_image():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLOv10Crack(
        num_classes=12,
        depth_multiple=0.33,
        width_multiple=0.25,
        use_attention=True
    ).to(device)

    pretrained_path = 'weights/yolov10n.pt'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        try:
            import torch.serialization
            with torch.serialization.add_safe_globals(['ultralytics.nn.tasks.YOLOv10DetectionModel']):
                checkpoint = torch.load(pretrained_path, map_location=device)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    optimizer = AdamW(model.parameters(), lr=0.001)
    
    dataset_yaml_path = os.path.join('data', 'dataset.yaml')
    
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
    
    model.train()
    
    with torch.no_grad():
        outputs = model(image)
        
    b, total_anchors, c = outputs.shape
    print(f"Total anchors: {total_anchors}")
    
    feature_map_sizes = [(64, 64), (32, 32), (16, 16)]
    strides = [8, 16, 32]
    
    all_anchor_centers = []
    anchor_idx = 0
    for level_idx, ((fh, fw), stride) in enumerate(zip(feature_map_sizes, strides)):
        for grid_y in range(fh):
            for grid_x in range(fw):
                cx = (grid_x + 0.5) * stride
                cy = (grid_y + 0.5) * stride
                all_anchor_centers.append((cx, cy, stride, anchor_idx))
                anchor_idx += 1
        print(f"Level {level_idx}: stride={stride}, fh={fh}, fw={fw}, anchors: {fh*fw}")
    
    print(f"Total anchor centers: {len(all_anchor_centers)}")
    
    target = labels[0]
    cls_id = int(target[0].item())
    x1, y1, x2, y2 = target[1].item(), target[2].item(), target[3].item(), target[4].item()
    gt_cx = (x1 + x2) / 2
    gt_cy = (y1 + y2) / 2
    
    print(f"\nTarget: class={cls_id}, box=[{x1}, {y1}, {x2}, {y2}], cx={gt_cx}, cy={gt_cy}")
    
    print("\nSearching for matching anchors...")
    matching_anchors = []
    for i, (cx, cy, stride, idx) in enumerate(all_anchor_centers):
        dist = ((gt_cx - cx) ** 2 + (gt_cy - cy) ** 2) ** 0.5
        threshold = stride * 2
        if dist < threshold:
            matching_anchors.append((idx, cx, cy, stride, dist))
    
    print(f"Found {len(matching_anchors)} matching anchors")
    for ma in matching_anchors[:20]:
        print(f"  Anchor {ma[0]}: cx={ma[1]:.1f}, cy={ma[2]:.1f}, stride={ma[3]}, dist={ma[4]:.1f}")

if __name__ == '__main__':
    train_single_image()
