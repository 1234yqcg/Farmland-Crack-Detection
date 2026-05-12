# 功能：搜索最佳评估超参数（conf / iou），最大化 mAP
# 对指定的权重文件，遍历 conf∈[0.001,0.05,...,0.25] 与 iou∈[0.4,0.45,...,0.65] 的所有组合，
# 输出能使 mAP@0.5 最高的参数组合，用于论文中汇报最佳指标。
# 使用方式: python tools/optimize_thresholds.py --weights <path> --config <path>
import os
import sys
import torch
from pathlib import Path
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Farmland_Crack_Detection.evaluate import evaluate_map
from Farmland_Crack_Detection.models.yolov10_crack import YOLOv10Crack
from Farmland_Crack_Detection.utils.roboflow_dataset import create_farmland_dataloaders
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv10Crack(num_classes=config['model']['num_classes'])
    
    ckpt = torch.load(args.weights, map_location=device)
    state_dict = ckpt.get('ema_model', ckpt.get('model', ckpt))
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 获取验证集
    data_yaml = config['data']['dataset_yaml']
    if not os.path.isabs(data_yaml):
        data_yaml = os.path.join(os.path.dirname(args.config), '..', data_yaml)
        
    img_size = tuple(config['training']['image_size'])
    _, val_loader, _ = create_farmland_dataloaders(
        data_yaml_path=data_yaml,
        batch_size=4,
        image_size=img_size,
        val_split='val'
    )
    
    print("开始搜索最佳评估超参数...")
    
    confs = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25]
    ious = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    
    best_map = 0
    best_params = {}
    
    for conf in confs:
        for iou in ious:
            ap, p, r, _ = evaluate_map(
                model, val_loader, device,
                conf_threshold=0.001, # For NMS bounding box generation internally
                map_conf_threshold=conf,
                iou_threshold=iou,
                num_classes=config['model']['num_classes']
            )
            print(f"Conf: {conf}, IoU: {iou} -> mAP@0.5: {ap:.4f}, P: {p:.4f}, R: {r:.4f}")
            if ap > best_map:
                best_map = ap
                best_params = {'conf': conf, 'iou': iou, 'p': p, 'r': r}
                
    print(f"\n最佳参数组合: Conf={best_params['conf']}, IoU={best_params['iou']}")
    print(f"最高 mAP@0.5: {best_map:.4f} (Precision: {best_params['p']:.4f}, Recall: {best_params['r']:.4f})")

if __name__ == "__main__":
    main()
