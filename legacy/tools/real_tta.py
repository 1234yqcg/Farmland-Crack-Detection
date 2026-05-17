#!/usr/bin/env python3
"""
Real Multi-Scale TTA for Custom YOLOv10
真正有效的测试时增强 - 多尺度 + 多方法
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack


class RealTTA:
    """
    真正的多尺度TTA实现
    
    包含：
    1. 多尺度推理 (640, 800, 1024, 1280)
    2. 水平翻转
    3. WBF (Weighted Boxes Fusion) 融合算法
    """
    
    def __init__(self, 
                 model: YOLOv10Crack,
                 device: torch.device,
                 num_classes: int = 3):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()
        
        # 多尺度列表（从小到大）
        self.scales = [640, 800, 1024]
        
    def _resize_image(self, image: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, float]:
        """调整图像大小并返回缩放比例"""
        _, _, h, w = image.shape
        
        # 计算缩放比例（保持长宽比）
        scale = min(target_size / w, target_size / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 使用双线性插值resize
        resized = torch.nn.functional.interpolate(
            image, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        return resized, scale
    
    def _pad_to_square(self, image: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """填充图像为正方形"""
        _, _, h, w = image.shape
        
        # 创建目标大小的画布
        padded = torch.zeros(1, 3, target_size, target_size, device=self.device)
        
        # 计算padding位置
        pad_x = (target_size - w) // 2
        pad_y = (target_size - h) // 2
        
        # 将图像粘贴到画布中心
        padded[:, :, pad_y:pad_y+h, pad_x:pad_x+w] = image
        
        return padded, (pad_x, pad_y)
    
    def _infer_single_scale(self, 
                             image: torch.Tensor, 
                             scale: int,
                             use_flip: bool = True) -> List[Dict]:
        """在单个尺度上进行推理（可选翻转）"""
        detections = []
        
        # Resize到目标尺度
        resized, resize_ratio = self._resize_image(image, scale)
        
        # 填充为正方形
        padded, (pad_x, pad_y) = self._pad_to_square(resized, scale)
        
        # 原始图像推理
        with torch.no_grad():
            outputs = self.model(padded)
            
        # 解码检测结果
        dets = self._decode_and_convert(outputs, scale, resize_ratio, (pad_x, pad_y))
        detections.extend(dets)
        
        # 翻转推理
        if use_flip:
            flipped = torch.flip(padded, dims=[3])
            
            with torch.no_grad():
                flip_outputs = self.model(flipped)
            
            flip_dets = self._decode_and_convert(
                flip_outputs, scale, resize_ratio, (pad_x, pad_y),
                is_flipped=True, original_width=resized.shape[-1]
            )
            detections.extend(flip_dets)
        
        return detections
    
    def _decode_and_convert(self, 
                            outputs: torch.Tensor,
                            scale: int,
                            resize_ratio: float,
                            padding: Tuple[int, int],
                            is_flipped: bool = False,
                            original_width: int = None) -> List[Dict]:
        """解码模型输出并转换回原图坐标系"""
        predictions = outputs.detach().cpu().numpy()
        
        if predictions.ndim == 3:
            predictions = predictions[0]
        
        if predictions.size == 0 or predictions.shape[1] < 5:
            return []
        
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        class_scores = predictions[:, 4:]  # 各类别得分
        
        if class_scores.size == 0:
            return []
        
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        
        detections = []
        pad_x, pad_y = padding
        
        for i in range(len(scores)):
            if scores[i] < 0.05:  # 低阈值，后面再过滤
                continue
            
            x1, y1, x2, y2 = boxes[i]
            
            # 移除padding偏移
            x1 -= pad_x
            x2 -= pad_x
            y1 -= pad_y
            y2 -= pad_y
            
            # 如果是翻转的，需要镜像回去
            if is_flipped and original_width:
                x1_new = original_width - x2
                x2_new = original_width - x1
                x1, y1, x2, y2 = x1_new, y1, x2_new, y2
            
            # 转换回原图尺寸
            x1 /= resize_ratio
            y1 /= resize_ratio
            x2 /= resize_ratio
            y2 /= resize_ratio
            
            # 过滤边界框
            if x2 > x1 and y2 > y1 and scores[i] > 0.01:
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(scores[i]),
                    'class_id': int(class_ids[i])
                })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            remaining = indices[1:]
            ious = self._compute_iou(boxes[i], boxes[remaining])
            indices = remaining[ious <= iou_threshold]
        
        return keep
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """计算IoU"""
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y2)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _wbf(self, detection_groups: List[List[Dict]], iou_threshold: float = 0.55) -> List[Dict]:
        """
        Weighted Boxes Fusion - 加权框融合
        比简单的NMS或平均融合效果更好
        """
        all_detections = []
        for group in detection_groups:
            all_detections.extend(group)
        
        if not all_detections:
            return []
        
        # 按类别分组
        by_class = {}
        for det in all_detections:
            cls_id = det['class_id']
            if cls_id not in by_class:
                by_class[cls_id] = []
            by_class[cls_id].append(det)
        
        final_detections = []
        
        for cls_id, cls_dets in by_class.items():
            if not cls_dets:
                continue
            
            boxes = np.array([d['bbox'] for d in cls_dets], dtype=np.float32)
            scores = np.array([d['score'] for d in cls_dets], dtype=np.float32)
            
            # WBF算法
            keep_indices = self._wbf_single_class(boxes, scores, iou_threshold)
            
            for idx in keep_indices:
                final_detections.append({
                    'bbox': boxes[idx].tolist(),
                    'score': float(scores[idx]),
                    'class_id': cls_id
                })
        
        # 按分数排序
        final_detections.sort(key=lambda x: x['score'], reverse=True)
        
        return final_detections
    
    def _wbf_single_class(self, 
                          boxes: np.ndarray, 
                          scores: np.ndarray, 
                          iou_threshold: float) -> List[int]:
        """单类别的WBF算法"""
        if len(boxes) == 0:
            return []
        
        # 按分数降序排列
        order = np.argsort(scores)[::-1]
        
        clusters = []
        used = set()
        
        for idx in order:
            if idx in used:
                continue
            
            # 创建新cluster
            cluster = [idx]
            used.add(idx)
            
            # 找所有与当前框重叠的框
            for other_idx in order:
                if other_idx in used:
                    continue
                
                iou = self._compute_iou(boxes[idx], boxes[other_idx:other_idx+1])[0]
                
                if iou >= iou_threshold:
                    cluster.append(other_idx)
                    used.add(other_idx)
            
            clusters.append(cluster)
        
        # 对每个cluster进行加权融合
        result_indices = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                result_indices.append(cluster[0])
            else:
                # 权重加权融合
                cluster_boxes = boxes[cluster]
                cluster_scores = scores[cluster]
                
                weight_sum = cluster_scores.sum()
                if weight_sum > 0:
                    fused_box = (cluster_boxes * cluster_scores[:, None]).sum(axis=0) / weight_sum
                    fused_score = cluster_scores.mean()
                    
                    # 用融合后的框替换第一个框的位置
                    first_idx = cluster[0]
                    boxes[first_idx] = fused_box
                    scores[first_idx] = fused_score
                    result_indices.append(first_idx)
        
        return result_indices
    
    def infer_with_tta(self, 
                       image: torch.Tensor, 
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45) -> List[Dict]:
        """
        使用完整TTA进行推理
        
        Args:
            image: 输入图像tensor [1, 3, H, W]
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            检测结果列表
        """
        all_detection_groups = []
        
        # 在多个尺度上推理
        for scale in self.scales:
            print(f"  TTA Scale: {scale}x{scale}")
            scale_dets = self._infer_single_scale(image, scale, use_flip=True)
            if scale_dets:
                all_detection_groups.append(scale_dets)
        
        if not all_detection_groups:
            return []
        
        # 使用WBF融合所有尺度的结果
        print("  Fusing multi-scale results...")
        final_detections = self._wbf(all_detection_groups, iou_threshold)
        
        # 最终NMS过滤
        if final_detections:
            boxes_array = np.array([d['bbox'] for d in final_detections])
            scores_array = np.array([d['score'] for d in final_detections])
            
            keep = self._nms(boxes_array, scores_array, iou_threshold)
            
            final_detections = [final_detections[i] for i in keep 
                               if final_detections[i]['score'] >= conf_threshold]
        
        return final_detections


def main():
    """测试TTA效果"""
    import argparse
    from PIL import Image
    import yaml
    
    parser = argparse.ArgumentParser(description='Real Multi-Scale TTA')
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--image', type=str, required=True, help='test image path')
    parser.add_argument('--data', type=str, default=None, help='dataset yaml')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集获取类别数
    num_classes = 3
    if args.data:
        with open(args.data, 'r') as f:
            cfg = yaml.safe_load(f)
        num_classes = cfg.get('nc', 3)
    
    # 加载模型
    print(f"Loading model from {args.weights}...")
    model = YOLOv10Crack(num_classes=num_classes).to(device)
    
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model', checkpoint.get('ema_model', checkpoint))
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 初始化TTA
    tta = RealTTA(model, device, num_classes)
    
    # 加载图像
    print(f"Loading image from {args.image}...")
    img = Image.open(args.image).convert('RGB')
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    # 普通推理
    print("\n[1] Normal inference (no TTA):")
    with torch.no_grad():
        normal_output = model(img_tensor)
    normal_dets = tta._decode_and_convert(normal_output, 1024, 1.0, (0, 0))
    normal_filtered = [d for d in normal_dets if d['score'] >= args.conf]
    print(f"  Detections: {len(normal_filtered)}")
    for d in normal_filtered[:5]:
        print(f"    Class {d['class_id']}: {d['score']:.3f} @ {d['bbox']}")
    
    # TTA推理
    print(f"\n[2] Multi-scale TTA (scales={tta.scales}):")
    tta_dets = tta.infer_with_tta(img_tensor, conf_threshold=args.conf)
    print(f"  Detections: {len(tta_dets)}")
    for d in tta_dets[:5]:
        print(f"    Class {d['class_id']}: {d['score']:.3f} @ {d['bbox']}")
    
    # 对比
    print(f"\n{'='*60}")
    print(f"Comparison:")
    print(f"  Normal: {len(normal_filtered)} detections")
    print(f"  TTA:     {len(tta_dets)} detections (+{len(tta_dets)-len(normal_filtered)})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
