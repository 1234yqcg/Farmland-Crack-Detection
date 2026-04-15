import os
import sys
import yaml
import torch
import cv2
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from utils.data_processing import ImagePreprocessor
from utils.visualization import ResultVisualizer

class CrackDetector:
    def __init__(self, 
                 model_path: str, 
                 data_yaml: str = None,
                 device: str = 'cuda',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 input_size: int = 512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.data_yaml = data_yaml or os.path.join(os.path.dirname(__file__), 'data', 'dataset.yaml')
        self.num_classes, self.class_names = self._load_data_config(self.data_yaml)
        self.class_colors = self._build_class_colors(self.num_classes)
        
        self.model = self._load_model(model_path)

        self.preprocessor = ImagePreprocessor(target_size=(input_size, input_size))
        self.visualizer = ResultVisualizer()
        self.visualizer.class_names = self.class_names
        self.visualizer.class_colors = self.class_colors

    def _load_data_config(self, data_yaml: str):
        with open(data_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        class_names = config.get('names', [])
        if isinstance(class_names, dict):
            class_names = [class_names[key] for key in sorted(class_names.keys(), key=int)]
        return int(config.get('nc', len(class_names) or 1)), {idx: name for idx, name in enumerate(class_names)}

    def _build_class_colors(self, num_classes: int):
        base_colors = [
            (0, 255, 0),
            (0, 255, 255),
            (0, 0, 255),
            (255, 0, 0),
            (255, 0, 255),
            (255, 255, 0),
        ]
        return {idx: base_colors[idx % len(base_colors)] for idx in range(num_classes)}
    
    def _load_model(self, model_path: str):
        model = YOLOv10Crack(num_classes=self.num_classes)
        
        if os.path.exists(model_path):
            checkpoint = None
            last_error = None
            for weights_only in (True, False):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=weights_only)
                    break
                except Exception as exc:
                    last_error = exc
            if checkpoint is None:
                raise last_error
            state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            model_dict = model.state_dict()
            compatible_state = {
                key: value for key, value in state_dict.items()
                if key in model_dict and model_dict[key].shape == value.shape
            }
            model_dict.update(compatible_state)
            model.load_state_dict(model_dict, strict=False)
        else:
            print(f"Warning: Model file not found at {model_path}")
        
        model.eval()
        model = model.to(self.device)
        return model

    def _box_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        if len(boxes) == 0:
            return []
        order = np.argsort(scores)[::-1]
        keep = []
        while len(order) > 0:
            idx = order[0]
            keep.append(int(idx))
            if len(order) == 1:
                break
            remaining = order[1:]
            ious = np.array([self._box_iou(boxes[idx], boxes[j]) for j in remaining])
            order = remaining[ious <= self.iou_threshold]
        return keep
    
    def postprocess(self, 
                    predictions: torch.Tensor,
                    transform_info: dict,
                    original_size: tuple) -> List[Dict]:
        predictions = predictions.cpu().numpy()
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        if predictions.size == 0:
            return []

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        if class_scores.size == 0:
            return []
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        keep_mask = scores >= self.conf_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        results = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            keep_indices = self._nms(class_boxes, class_scores)
            for idx in keep_indices:
                x1, y1, x2, y2 = class_boxes[idx]
                x1 = int((x1 - transform_info['pad_x']) / transform_info['scale'])
                y1 = int((y1 - transform_info['pad_y']) / transform_info['scale'])
                x2 = int((x2 - transform_info['pad_x']) / transform_info['scale'])
                y2 = int((y2 - transform_info['pad_y']) / transform_info['scale'])
                x1 = max(0, min(original_size[0], x1))
                y1 = max(0, min(original_size[1], y1))
                x2 = max(0, min(original_size[0], x2))
                y2 = max(0, min(original_size[1], y2))
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(class_scores[idx]),
                    'class_id': int(class_id),
                    'class_name': self.class_names.get(int(class_id), str(class_id)),
                    'color': self.class_colors.get(int(class_id), (255, 255, 255))
                })

        results.sort(key=lambda item: item['score'], reverse=True)
        return results
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Dict]:
        if image is None:
            return []
        
        original_size = (image.shape[1], image.shape[0])
        
        processed_image, transform_info = self.preprocessor.preprocess_pipeline(image)
        
        input_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.float().div(255.0).to(self.device)
        
        predictions = self.model(input_tensor)
        
        results = self.postprocess(predictions, transform_info, original_size)
        
        return results
    
    def detect_and_visualize(self, image: np.ndarray) -> tuple:
        results = self.detect(image)
        result_image = self.visualizer.draw_detections(image, results)
        return result_image, results

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                       help='path to input image')
    parser.add_argument('--model', type=str, default='weights/best_model.pt',
                       help='path to model weights')
    parser.add_argument('--data', type=str, default=None,
                       help='path to dataset yaml')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='path to output image')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='nms IoU threshold')
    parser.add_argument('--img-size', type=int, default=512,
                       help='input image size')
    args = parser.parse_args()
    
    detector = CrackDetector(
        model_path=args.model,
        data_yaml=args.data,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.img_size
    )
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Cannot load image from {args.image}")
        return
    
    result_image, results = detector.detect_and_visualize(image)
    
    cv2.imwrite(args.output, result_image)
    
    print(f"Detection complete!")
    print(f"Found {len(results)} cracks")
    print(f"Result saved to {args.output}")

if __name__ == '__main__':
    main()
