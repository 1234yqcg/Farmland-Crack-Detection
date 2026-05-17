import os
import torch
import numpy as np
from typing import List, Tuple, Union, Optional
from pathlib import Path

from models.yolov10_crack import YOLOv10Crack


def build_model(weight_path: str, 
                device: torch.device,
                num_classes: int = 3) -> Tuple[YOLOv10Crack, dict]:
    """
    构建并加载模型
    
    Args:
        weight_path: 模型权重路径
        device: 计算设备
        num_classes: 类别数量
        
    Returns:
        (模型实例, 配置信息字典)
    """
    model = YOLOv10Crack(num_classes=num_classes)
    
    config = {
        'weight_path': weight_path,
        'num_classes': num_classes,
        'device': str(device)
    }
    
    # 加载权重（如果存在）
    if os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            config['loaded'] = True
            print(f"✓ 成功加载模型权重: {weight_path}")
        except Exception as e:
            print(f"⚠ 加载权重失败，使用随机初始化: {e}")
            config['loaded'] = False
    else:
        print(f"⚠ 权重文件不存在，使用随机初始化: {weight_path}")
        config['loaded'] = False
    
    model = model.to(device)
    model.eval()
    
    return model, config


def parse_weight_paths(weights_str: str) -> List[str]:
    """
    解析模型权重路径字符串
    
    支持多个权重（逗号分隔），用于模型集成
    
    Args:
        weights_str: 权重路径或逗号分隔的多个路径
        
    Returns:
        权重路径列表
    """
    if not weights_str:
        return []
    
    paths = [w.strip() for w in weights_str.split(',')]
    return [p for p in paths if p]


def resolve_default_data_yaml() -> str:
    """
    解析默认的data.yaml文件路径
    
    Returns:
        data.yaml文件的绝对路径
    """
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 常见的数据集配置位置
    candidates = [
        os.path.join(current_dir, 'data', 'data.yaml'),
        os.path.join(current_dir, '..', 'data', 'data.yaml'),
        os.path.join(os.getcwd(), 'data', 'data.yaml')
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    
    # 返回默认路径（即使不存在）
    return os.path.join(current_dir, 'data', 'data.yaml')


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_map(models: Union[YOLOv10Crack, List[YOLOv10Crack]],
                 dataloader,
                 device: torch.device,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 num_classes: int = 3) -> Tuple[float, float, float, dict]:
    """
    评估模型的mAP指标
    
    Args:
        models: 单个模型或模型列表（用于集成）
        dataloader: 数据加载器
        device: 计算设备
        conf_threshold: 置信度阈值
        iou_threshold: NMS的IoU阈值
        num_classes: 类别数量
        
    Returns:
        (mAP, precision, recall, AP_per_class字典)
    """
    if not isinstance(models, list):
        models = [models]
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            targets = batch['labels'].cpu().numpy()
            
            batch_size = images.size(0)
            
            for model in models:
                outputs = model(images)
                
                for i in range(batch_size):
                    pred = outputs[i].cpu().numpy()
                    
                    # 解码预测结果
                    if pred.ndim == 2 and pred.shape[1] >= 5:
                        scores = pred[:, 4]  # 置信度
                        
                        # 过滤低置信度预测
                        high_conf_mask = scores > conf_threshold
                        high_conf_preds = pred[high_conf_mask]
                        
                        for p in high_conf_preds:
                            cx, cy, w, h = p[0:4]
                            conf = p[4]
                            
                            # 转换为xyxy格式
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            
                            class_id = int(np.argmax(p[5:5+num_classes])) if p.shape[0] > 5 else 0
                            
                            all_predictions.append({
                                'box': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_id': class_id,
                                'image_id': batch_idx * batch_size + i
                            })
                    
                    # 收集真值标注
                    gt_labels = targets[i]
                    for label in gt_labels:
                        if label[0] != -1:  # 忽略无目标标记
                            class_id = int(label[0])
                            cx, cy, w, h = label[1:5]
                            
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            
                            all_ground_truths.append({
                                'box': [x1, y1, x2, y2],
                                'class_id': class_id,
                                'image_id': batch_idx * batch_size + i
                            })
    
    # 计算mAP
    ap_per_class = {}
    total_ap = 0.0
    valid_classes = 0
    
    for cls_id in range(num_classes):
        # 获取该类别的预测和真值
        cls_preds = [p for p in all_predictions if p['class_id'] == cls_id]
        cls_gts = [g for g in all_ground_truths if g['class_id'] == cls_id]
        
        if len(cls_gts) == 0:
            continue
        
        # 按置信度排序预测
        cls_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 计算AP
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        
        gt_matched = set()
        
        for i, pred in enumerate(cls_preds):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(cls_gts):
                if j in gt_matched:
                    continue
                
                if pred['image_id'] != gt['image_id']:
                    continue
                
                iou = calculate_iou(
                    np.array(pred['box']),
                    np.array(gt['box'])
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # 计算precision和recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(cls_gts)
        
        # 计算AP（11点插值）
        ap = _compute_ap(precision, recall)
        ap_per_class[cls_id] = ap
        total_ap += ap
        valid_classes += 1
    
    mAP = total_ap / valid_classes if valid_classes > 0 else 0.0
    
    # 计算整体precision和recall
    total_tp = sum(1 for p in all_predictions if any(
        calculate_iou(np.array(p['box']), np.array(g['box'])) >= iou_threshold 
        and p['image_id'] == g['image_id']
        and p['class_id'] == g['class_id']
        for g in all_ground_truths
    ))
    
    precision = total_tp / len(all_predictions) if all_predictions else 0.0
    recall = total_tp / len(all_ground_truths) if all_ground_truths else 0.0
    
    return mAP, precision, recall, ap_per_class


def _compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    使用11点插值法计算AP
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask])
    
    return ap / 11.0
