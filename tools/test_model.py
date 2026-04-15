import os
import sys
import yaml
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from utils.roboflow_dataset import RoboflowFarmlandDataset

def test_inference():
    model_path = 'outputs/exp_anchor_loss/weights/test_model.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = YOLOv10Crack(
        num_classes=1,
        depth_multiple=0.33,
        width_multiple=0.25,
        use_attention=True
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    dataset_yaml_path = r'd:\桌面\学习\000毕业设计！\YOLOv10\farmland_crack_detection\data\dataset.yaml'
    
    dataset = RoboflowFarmlandDataset(
        data_yaml_path=dataset_yaml_path,
        split='train',
        image_size=(512, 512),
        augment=False,
        cache_images=False
    )
    
    print(f"Total samples: {len(dataset)}")
    
    data = dataset[0]
    image_tensor = data['image'].unsqueeze(0).to(device)
    labels = data['labels']
    original_h, original_w = 512, 512
    
    print(f"Image shape: {image_tensor.shape}")
    print(f"Labels: {labels}")
    
    transform_info = {
        'pad_x': 0,
        'pad_y': 0,
        'scale': 1.0
    }
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    predictions = outputs[0].cpu().numpy()
    print(f"Predictions shape: {predictions.shape}")
    
    num_classes = 1
    num_anchors = 16
    
    class_scores_all = predictions[:, 4*num_anchors:]
    reg_predictions_full = predictions[:, :4*num_anchors]
    
    target_reg_channels = 60
    reg_predictions = reg_predictions_full[:, :target_reg_channels]
    
    reg_channels = reg_predictions.shape[1]
    reg_max = reg_channels // 4
    print(f"Using reg_max: {reg_max}")
    
    exp_scores = np.exp(class_scores_all - np.max(class_scores_all, axis=1, keepdims=True))
    class_scores_softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    print(f"Class scores - min: {class_scores_softmax.min():.6f}, max: {class_scores_softmax.max():.6f}")
    
    reg_pred_reshaped = reg_predictions.reshape(-1, 4, reg_max)
    exp_reg = np.exp(reg_pred_reshaped - np.max(reg_pred_reshaped, axis=2, keepdims=True))
    reg_pred_decoded = (exp_reg * np.arange(reg_max, dtype=np.float32)).sum(axis=2)
    
    strides = [8, 16, 32]
    feature_map_sizes = [(64, 64), (32, 32), (16, 16)]
    
    anchor_sizes = [
        (10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198),
        (373, 326), (10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90)
    ]
    
    results = []
    pred_idx = 0
    
    for level_idx, (stride, (fh, fw)) in enumerate(zip(strides, feature_map_sizes)):
        for grid_y in range(fh):
            for grid_x in range(fw):
                for anchor_idx in range(num_anchors):
                    if pred_idx >= len(predictions):
                        break
                    
                    class_scores = class_scores_softmax[pred_idx]
                    class_id = int(np.argmax(class_scores))
                    score = float(np.max(class_scores))
                    
                    reg_pred = reg_pred_decoded[pred_idx]
                    
                    if score >= 0.3:
                        anchor_w, anchor_h = anchor_sizes[anchor_idx]
                        
                        cx = (grid_x + 0.5) * stride
                        cy = (grid_y + 0.5) * stride
                        
                        x1 = cx - reg_pred[0] * anchor_w / reg_max
                        y1 = cy - reg_pred[1] * anchor_h / reg_max
                        x2 = cx + reg_pred[2] * anchor_w / reg_max
                        y2 = cy + reg_pred[3] * anchor_h / reg_max
                        
                        x1 = max(0, min(original_w, int(x1)))
                        y1 = max(0, min(original_h, int(y1)))
                        x2 = max(0, min(original_w, int(x2)))
                        y2 = max(0, min(original_h, int(y2)))
                        
                        if pred_idx < 10:
                            print(f"  pred[{pred_idx}] - class_id: {class_id}, score: {score:.6f}, bbox: [{x1}, {y1}, {x2}, {y2}]")
                        
                        results.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': float(score),
                            'class_id': class_id,
                            'class_name': 'Crack'
                        })
                    
                    pred_idx += 1
    
    def nms(boxes, scores, iou_threshold=0.5):
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    if len(results) > 0:
        boxes = [r['bbox'] for r in results]
        scores = [r['score'] for r in results]
        
        keep_indices = nms(boxes, scores, iou_threshold=0.3)
        
        results = [results[i] for i in keep_indices]
    
    print(f"\nTotal detections after NMS: {len(results)}")
    
    for i, det in enumerate(results):
        print(f"  Detection {i+1}: bbox={det['bbox']}, score={det['score']:.4f}")

if __name__ == '__main__':
    test_inference()
