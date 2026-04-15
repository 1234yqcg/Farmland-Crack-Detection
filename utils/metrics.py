import numpy as np
from typing import List, Dict, Optional

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-7)

def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate([[0], recalls, [1]])
    mpre = np.concatenate([[0], precisions, [0]])
    
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    
    return ap

def calculate_map(predictions: List[Dict], 
                  targets: List[Dict],
                  iou_thresholds: List[float] = None,
                  num_classes: Optional[int] = None) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    
    results = {}
    
    if num_classes is None:
        class_ids = set()
        for pred, target in zip(predictions, targets):
            class_ids.update(p['class_id'] for p in pred)
            class_ids.update(t['class_id'] for t in target)
        num_classes = max(class_ids) + 1 if class_ids else 1
    
    for iou_thresh in iou_thresholds:
        aps = []
        
        for class_id in range(num_classes):
            class_preds = []
            class_targets = []
            
            for image_idx, (pred, target) in enumerate(zip(predictions, targets)):
                class_preds.extend([
                    (image_idx, p['score'], p['bbox']) 
                    for p in pred if p['class_id'] == class_id
                ])
                class_targets.extend([
                    (image_idx, t['bbox']) 
                    for t in target if t['class_id'] == class_id
                ])
            
            if not class_targets:
                continue
            
            class_preds.sort(key=lambda x: x[1], reverse=True)
            
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            matched = set()
            
            for i, (pred_image_idx, score, pred_bbox) in enumerate(class_preds):
                best_iou = 0
                best_idx = -1
                
                for j, (target_image_idx, target_bbox) in enumerate(class_targets):
                    if j in matched or target_image_idx != pred_image_idx:
                        continue
                    
                    iou = calculate_iou(np.array(pred_bbox), np.array(target_bbox))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                
                if best_iou >= iou_thresh:
                    tp[i] = 1
                    matched.add(best_idx)
                else:
                    fp[i] = 1
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(class_targets)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            ap = calculate_ap(recalls, precisions)
            aps.append(ap)
        
        results[f'mAP@{iou_thresh}'] = np.mean(aps) if aps else 0.0
    
    return results

def calculate_precision_recall(predictions: List[Dict], 
                               targets: List[Dict],
                               iou_threshold: float = 0.5) -> Dict[str, float]:
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for pred, target in zip(predictions, targets):
        matched = set()
        
        for p in pred:
            best_iou = 0
            best_idx = -1
            
            for i, t in enumerate(target):
                if i in matched:
                    continue
                
                iou = calculate_iou(np.array(p['bbox']), np.array(t['bbox']))
                if iou > best_iou and p['class_id'] == t['class_id']:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_threshold:
                all_tp += 1
                matched.add(best_idx)
            else:
                all_fp += 1
        
        all_fn += len(target) - len(matched)
    
    precision = all_tp / (all_tp + all_fp + 1e-7)
    recall = all_tp / (all_tp + all_fn + 1e-7)
    
    return {
        'precision': precision,
        'recall': recall,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn
    }
