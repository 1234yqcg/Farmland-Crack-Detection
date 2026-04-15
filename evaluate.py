import os
import sys
import warnings
warnings.filterwarnings('ignore')

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.yolov10_crack import YOLOv10Crack
from utils.metrics import calculate_map, calculate_precision_recall
from utils.roboflow_dataset import RoboflowFarmlandDataset


def box_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    box2_area = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.5):
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
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in remaining])
        indices = remaining[ious <= iou_threshold]
    return keep


def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        images.append(item['image'])
        targets.append(item['labels'])
    images = torch.stack(images, dim=0)
    return {'images': images, 'targets': targets}


def load_data_config(dataset_yaml):
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_default_data_yaml():
    return os.path.join(os.path.dirname(__file__), 'data', 'dataset.yaml')


def split_exists(data_yaml, data_config, split):
    split_path = data_config.get(split)
    if not split_path:
        return False
    if not os.path.isabs(split_path):
        split_path = os.path.join(os.path.dirname(data_yaml), split_path)
    if not os.path.exists(split_path):
        return False
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return any(name.lower().endswith(image_extensions) for name in os.listdir(split_path))


def build_model(weights_path, device, num_classes):
    model = YOLOv10Crack(num_classes=num_classes, reg_max=16).to(device)
    if os.path.exists(weights_path):
        checkpoint = None
        last_error = None
        for weights_only in (True, False):
            try:
                checkpoint = torch.load(weights_path, map_location=device, weights_only=weights_only)
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
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: weights not found at {weights_path}")
    model.eval()
    return model


def decode_predictions(predictions, conf_threshold=0.5, iou_threshold=0.5):
    predictions = predictions.detach().cpu().numpy()
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.size == 0:
        return []
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    if class_scores.size == 0:
        return []
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    keep_mask = scores >= conf_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    detections = []
    for class_id in np.unique(class_ids):
        cls_mask = class_ids == class_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep_indices = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep_indices:
            detections.append({
                'bbox': cls_boxes[idx].tolist(),
                'score': float(cls_scores[idx]),
                'class_id': int(class_id)
            })
    detections.sort(key=lambda item: item['score'], reverse=True)
    return detections


def evaluate_map(model, dataloader, device, conf_threshold=0.5, iou_threshold=0.5, num_classes=1):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            targets = batch['targets']
            outputs = model(images)
            for sample_idx in range(outputs.shape[0]):
                detections = decode_predictions(outputs[sample_idx], conf_threshold, iou_threshold)
                sample_targets = []
                target = targets[sample_idx]
                if len(target) > 0:
                    for t in target:
                        _, class_id, x1, y1, x2, y2 = t.tolist()
                        sample_targets.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': int(class_id)
                        })
                all_predictions.append(detections)
                all_targets.append(sample_targets)
    map_results = calculate_map(all_predictions, all_targets, iou_thresholds=[iou_threshold], num_classes=num_classes)
    pr_results = calculate_precision_recall(all_predictions, all_targets, iou_threshold=iou_threshold)
    return map_results.get(f'mAP@{iou_threshold}', 0.0), pr_results['precision'], pr_results['recall']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='outputs/exp_anchor_loss/weights/best.pt', help='path to model weights')
    parser.add_argument('--data', type=str, default=resolve_default_data_yaml(), help='path to dataset yaml')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS and metrics')
    parser.add_argument('--img-size', type=int, default=512, help='input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='evaluation batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_config = load_data_config(args.data)
    num_classes = int(data_config['nc'])
    model = build_model(args.weights, device, num_classes)

    val_split = 'val' if split_exists(args.data, data_config, 'val') else 'test'
    val_dataset = RoboflowFarmlandDataset(args.data, val_split, (args.img_size, args.img_size), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"\nEvaluating on {val_split} set ({len(val_dataset)} images)...")
    ap, precision, recall = evaluate_map(
        model,
        val_loader,
        device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        num_classes=num_classes
    )

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results (conf={args.conf}, iou={args.iou})")
    print(f"{'=' * 50}")
    print(f"mAP@0.5: {ap:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"{'=' * 50}")
